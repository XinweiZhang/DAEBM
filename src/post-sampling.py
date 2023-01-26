import argparse
import json
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.config import add_daebm_parser
    from lib.diffusion import MGMS_sampling, make_sigma_schedule
    from lib.train import configure_net
    from lib.utils import (Accumulator, imshow, inv_data_transform,
                           make_figure_grid)
except ImportError:
    raise

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main(args):

    pexp_dir = os.path.join(args.main_dir, args.pexp_dir)
    exp_dir = os.path.join(pexp_dir, "post_sampling_sim")

    # create experiment folder if not exists
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)

    os.makedirs(os.path.join(exp_dir, "TravelTime"))
    os.makedirs(os.path.join(exp_dir, "Images"))
    os.makedirs(os.path.join(exp_dir, "Time0Images"))
    os.makedirs(os.path.join(exp_dir, "Time0ImagesSim"))

    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("DAEBM (Post Sampling)")
    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(filename=f"{exp_dir}/plog.txt", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    sys.stderr = open(f"{exp_dir}/perr.txt", "w")

    torch.manual_seed(args.t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.t_seed)
    np.random.seed(args.t_seed)

    "Networks Configuration"
    net = configure_net(args)
    logger.info(str(net))

    checkpoint = torch.load(
        pexp_dir + "/saved_models/net_checkpoint.pt", map_location=device
    )
    net.load_state_dict(checkpoint["state_dict"])
    net = net.to(device)

    step_size = checkpoint["step_size"]

    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt = make_sigma_schedule(
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1],
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        schedule=args.diffusion_schedule,
    )
    logger.info("load checkpoint at epoch {}.".format(checkpoint["epoch"]))
    logger.info("sigmas:" + str(sigmas))
    logger.info("alphas:" + str(alphas))
    logger.info("alpha_bar_sqrt:" + str(alphas_bar_sqrt))
    logger.info("alphas_bar_comp_sqrt:" + str(alphas_bar_comp_sqrt))

    logger.info("step_size:" + str(step_size))

    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt, step_size = (
        sigmas.to(device),
        alphas.to(device),
        alphas_bar_sqrt.to(device),
        alphas_bar_comp_sqrt.to(device),
        step_size.to(device),
    )

    is_reject = (
        True if (args.mala_reject is True or args.dynamic_sampling is True) else False
    )

    time0_bank = torch.randn((0,) + tuple(args.image_shape))

    sampling_chains = args.sampling_chains
    accumulators = {}
    accumulators["mala_acpt_rate"] = Accumulator(args.num_diffusion_timesteps + 1)
    accumulators["labels"] = Accumulator(
        sampling_chains * (args.num_diffusion_timesteps + 1)
    )
    accumulators["labels_jump"] = Accumulator(sampling_chains)
    accumulators["labels_jump_mat"] = Accumulator(
        (args.num_diffusion_timesteps + 1) ** 2
    )
    accumulators["labels_cum_jump_mat"] = Accumulator(
        (args.num_diffusion_timesteps + 1) ** 2
    )

    init_x_t_neg = torch.randn((sampling_chains,) + tuple(args.image_shape))
    init_t_neg = torch.ones(sampling_chains).fill_(args.num_diffusion_timesteps).long()

    count0 = torch.zeros_like(init_t_neg)
    mark0 = torch.zeros_like(init_t_neg)
    traverse_time = torch.zeros(sampling_chains)
    traverse_time_bank = torch.zeros(0)
    for n_iter in range(args.total_iters):
        x_t_neg, t_neg, acpt_rate, _ = MGMS_sampling(
            net,
            init_x_t_neg.to(device),
            init_t_neg.to(device),
            args.sample_steps,
            init_step_size=step_size,
            reject=is_reject,
        )

        accumulators["mala_acpt_rate"].add(1, acpt_rate.nan_to_num())

        count0 += t_neg == 0

        time0_samples_idx = torch.logical_and(count0 == args.stop_a_chain_M, mark0 == 0)
        mark0[time0_samples_idx] = torch.ones_like(mark0[time0_samples_idx])

        time0_samples = x_t_neg[time0_samples_idx]
        if args.renew_chains:
            x_t_neg[time0_samples_idx] = torch.randn_like(time0_samples)

        traverse_time += args.sample_steps
        time0_traverse_time = traverse_time[time0_samples_idx]
        if args.renew_chains:
            traverse_time[time0_samples_idx] = torch.zeros(len(time0_samples))
            mark0[time0_samples_idx] = torch.zeros_like(mark0[time0_samples_idx])
            count0[time0_samples_idx] = torch.zeros_like(mark0[time0_samples_idx])

        time0_bank = torch.cat([time0_bank, time0_samples], 0)
        traverse_time_bank = torch.cat([traverse_time_bank, time0_traverse_time], 0)

        init_x_t_neg, init_t_neg = x_t_neg, t_neg

        if (n_iter + 1) % args.print_freq == 0:
            logger.info(f"Iterations:{n_iter}")
            if len(time0_samples) != 0:
                logger.info(
                    "acceptance rate: {acpt_rate:.4f} "
                    "pixels mean: {pixels_mean:.4f} "
                    "pixels min: {pixels_min:.4f} "
                    "pixels max: {pixels_max:.4f} ".format(
                        acpt_rate=torch.mean(
                            torch.tensor(accumulators["mala_acpt_rate"].average())
                        ),
                        pixels_mean=time0_samples.mean().item(),
                        pixels_min=time0_samples.min().item(),
                        pixels_max=time0_samples.max().item(),
                    )
                )
            logger.info(
                "AcptRate "
                + "".join(
                    [
                        f"t_{i}:{acpt_rate[i]:.2f}; "
                        for i in range((args.num_diffusion_timesteps + 1))
                    ]
                )
            )

            if len(time0_bank) >= 100:
                figure = make_figure_grid(
                    inv_data_transform(time0_bank[-100:], args.data_transform),
                    ncol=10,
                    nrow=10,
                    figsize=(8, 8),
                    show=False,
                )
                figure.savefig(
                    exp_dir + f"/Time0Images/Time0Images_At_{n_iter + 1}.png"
                )
            if len(time0_bank) >= 20:
                imshow(
                    inv_data_transform(time0_bank[-20:], args.data_transform),
                    ncol=5,
                    save_dir=exp_dir
                    + f"/Time0ImagesSim/Time0Images_At_{n_iter + 1}.png",
                )

            figure = make_figure_grid(
                inv_data_transform(x_t_neg[:100], args.data_transform),
                t_neg,
                ncol=10,
                nrow=10,
                figsize=(8, 8),
                show=False,
            )
            figure.savefig(exp_dir + f"/Images/Images_At_{n_iter + 1}.png")

        if (n_iter + 1) % 2000 == 0:

            if len(traverse_time_bank) > 0:
                fig = plt.figure()
                plt.hist(traverse_time_bank.numpy())
                fig.savefig(exp_dir + f"/TravelTime/TravelTime_At_{n_iter + 1}.png")
                plt.close()
                np.savetxt(
                    exp_dir + "/TravelTime/TravelTime.csv",
                    traverse_time_bank.numpy(),
                    fmt="%d",
                    delimiter=",",
                )

            sample_dict = {
                "samples": x_t_neg,
                "labels": t_neg,
            }
            torch.save(sample_dict, exp_dir + f"/samples_{n_iter}.pt")

        time0_bank_dict = {
            "samples": time0_bank,
        }
        torch.save(time0_bank_dict, exp_dir + "/time0_samples.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    cli_parser = argparse.ArgumentParser(
        description="configuration arguments provided at run time from the CLI"
    )

    cli_parser.add_argument(
        "-hm",
        "--help_more",
        dest="help_more",
        action="store_true",
        default=False,
        help="more help on the running parameters",
    )

    cli_parser.add_argument(
        "--main_dir", type=str, default="./svhn", help="directory of the experiment"
    )

    cli_parser.add_argument(
        "--pexp_dir", type=str, default=None, help="directory of the plot experiment"
    )

    cli_parser.add_argument(
        "--total_iters",
        type=int,
        default=10000,
        help="total iterations/transitions of MGMS",
    )

    cli_parser.add_argument(
        "--sampling_chains", type=int, default=100, help="sampling chains in parallel"
    )

    cli_parser.add_argument(
        "--stop_a_chain_M",
        type=int,
        default=50,
        help="stopping creterion when reach 0 for M times",
    )

    cli_parser.add_argument(
        "--renew_chains", default=False, action="store_true", help="whether renew chain"
    )
    args, unknown = cli_parser.parse_known_args()
    add_parser = add_daebm_parser()
    parser = argparse.ArgumentParser(parents=[cli_parser, add_parser], add_help=False)

    if args.help_more:
        parser.print_help()

    pexp_dir = os.path.join(args.main_dir, args.pexp_dir)

    if pexp_dir is not None:
        with open(pexp_dir + "/hparams.json", "r") as myfile:
            data = myfile.read()
            hparams = json.loads(data)

        parser.set_defaults(**hparams)

        args = parser.parse_args()

        main(args)
