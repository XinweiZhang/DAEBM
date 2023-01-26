import argparse
import json
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.config import add_daebm_parser
    from lib.diffusion import (MSGS_sampling, adjust_step_size_given_acpt_rate,
                               make_sigma_schedule, q_sample,
                               q_sample_progressive)
    from lib.libtoy import (ToyDataset, plot_decision_surface,
                            plot_diffusion_densities_along_a_line)
    from lib.sampler import ReplayBuffer
    from lib.train import (configure_net, configure_optimizer,
                           configure_scheduler, initialize_net,
                           save_checkpoint)
    from lib.utils import Accumulator, AverageMeter
    from lib.write_tb import write_results
except ImportError:
    raise


# === Training loss ===
def training_losses(net, x_pos, t, x_neg, t_neg):
    """
    Training loss calculation
    """
    pos_f = net.energy_output(x_pos, t)
    neg_f = net.energy_output(x_neg, t_neg)
    loss = (pos_f - neg_f).squeeze().mean()

    return loss


def main(args):

    exp_dir = os.path.join(args.main_dir, args.exp_dir)
    # data_dir = os.path.join(args.main_dir, args.data_dir)

    # create experiment folder if not exists
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        if args.refresh:
            shutil.rmtree(exp_dir)
            os.makedirs(exp_dir)
        else:
            raise RuntimeError("Exp Folder Exists")

    kwargs = vars(args)
    with open(os.path.join(exp_dir, "init_hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=False, indent=4)

    sys.stderr = open(f"{exp_dir}/err.txt", "w")
    os.makedirs(exp_dir + "/jump_mat")

    log_dir = None if args.log_dir == "None" else os.path.join(exp_dir, args.log_dir)

    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("Toy DA-EBM Training")
    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(filename=f"{exp_dir}/log.txt", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    saved_models_dir = (
        None
        if args.saved_models_dir == "None"
        else os.path.join(exp_dir, args.saved_models_dir)
    )

    writer = SummaryWriter(log_dir=log_dir) if args.log_dir is not None else None

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.t_seed)
    np.random.seed(args.t_seed)

    q = ToyDataset(
        args.toy_type,
        args.toy_groups,
        args.toy_sd,
        args.toy_radius,
        args.viz_res,
        args.kde_bw,
    )

    (x_, y_) = q.sample_toy_data(50000)
    x_ = torch.FloatTensor(x_)
    y_ = torch.zeros(x_.shape[0]).long()
    train_set = TensorDataset(x_, y_)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    args.n_class = args.num_diffusion_timesteps + 1
    "Networks Configuration"
    net = configure_net(args)
    logger.info(str(net))
    net = initialize_net(net, args.net_init_method).to(device)

    "Optimizer Configuration"
    optimizer = configure_optimizer(
        net.parameters(),
        args.optimizer_type,
        args.lr,
        args.weight_decay,
        args.betas,
        args.sgd_momentum,
    )

    scheduler, warmup_scheduler = configure_scheduler(
        optimizer,
        args.scheduler_type,
        args.milestones,
        args.lr_decay_factor,
        args.n_epochs,
        n_warm_iters=args.n_warm_epochs * len(train_loader),
    )

    "Sampling Configuration"
    is_reject = args.mala_reject
    replay_buffer = ReplayBuffer(
        buffer_size=args.replay_buffer_size,
        image_shape=(2,),
        n_class=args.num_diffusion_timesteps + 1,
        random_image_type=args.random_image_type,
    )

    "Diffusion Configuration"
    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt = make_sigma_schedule(
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1],
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        schedule=args.diffusion_schedule,
    )

    b_square = args.b_factor ** 2
    step_size_square = sigmas ** 2 * b_square
    step_size_square[0] = step_size_square[1]

    init_step_size = step_size_square.sqrt()

    logger.info("sigmas:" + str(sigmas))
    logger.info("alphas:" + str(alphas))
    logger.info("alpha_bar_sqrt:" + str(alphas_bar_sqrt))
    logger.info("alphas_bar_comp_sqrt:" + str(alphas_bar_comp_sqrt))

    logger.info("step_size:" + str(init_step_size))

    train_set_fig = plot_decision_surface(
        q,
        None,
        train_set[:][0].squeeze(),
        labels=None,
        device=device,
        save_path="return",
    )
    writer.add_figure("Train Set", train_set_fig, global_step=0)
    diffuse_data_pool = q_sample_progressive(
        x_[torch.randperm(x_.shape[0])[:5000]], alphas, sigmas
    )
    fig, axes = plt.subplots(1, args.num_diffusion_timesteps + 1, figsize=(21, 3))
    for i in range(args.num_diffusion_timesteps + 1):
        diffuse_data = diffuse_data_pool[i]
        axes[i].scatter(*(diffuse_data.T))
    plt.tight_layout()
    # plt.show()
    writer.add_figure("Diffusion Data Figure", fig, global_step=0)
    plt.close()

    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt, init_step_size = (
        sigmas.to(device),
        alphas.to(device),
        alphas_bar_sqrt.to(device),
        alphas_bar_comp_sqrt.to(device),
        init_step_size.to(device),
    )

    "Ancillary"
    n_iter = 0
    time0_bank = torch.randn((50000,) + (2,))
    fid_save_sample_idx = 0

    accumulators = {}
    accumulators["mala_acpt_rate"] = Accumulator(args.num_diffusion_timesteps + 1)
    accumulators["labels_jump_mat"] = Accumulator(
        (args.num_diffusion_timesteps + 1) ** 2
    )

    meter_list = ["loss"]
    x_0 = next(iter(train_loader))[0]
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}

        for idx, (x_0, _) in enumerate(train_loader):
            n_iter += 1

            t = torch.randint(
                high=args.num_diffusion_timesteps + 1, size=(x_0.shape[0],)
            ).to(device)

            x_t = q_sample(x_0.to(device), t, alphas_bar_sqrt, alphas_bar_comp_sqrt)

            init_x_t_neg, init_t_neg, buffer_idx = replay_buffer.sample_buffer(
                n_samples=args.batch_size, reinit_probs=args.reinit_probs,
            )

            x_t_neg, t_neg, acpt_rate, _ = MSGS_sampling(
                net,
                init_x_t_neg.to(device),
                init_t_neg.to(device),
                args.sample_steps,
                init_step_size=init_step_size,
                reject=is_reject,
            )

            accumulators["mala_acpt_rate"].add(1, acpt_rate.nan_to_num())

            jump_mat = np.zeros(
                ((args.num_diffusion_timesteps + 1), (args.num_diffusion_timesteps + 1))
            )
            jump_coordinates = (
                torch.cat([init_t_neg.view(-1, 1), t_neg.view(-1, 1)], 1).cpu().numpy()
            )
            np.add.at(jump_mat, tuple(zip(*jump_coordinates)), 1)
            accumulators["labels_jump_mat"].add(1, jump_mat.reshape(-1))

            t_neg = t_neg.to(device)
            x_t_neg = x_t_neg.to(device)

            loss = training_losses(net, x_t, t, x_t_neg, t_neg)

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                logger.error("Training breakdown")
                args.breakdown = "Breakdown"
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < args.n_warm_epochs:
                warmup_scheduler.step()

            image_samples = x_t_neg.cpu()
            image_labels = t_neg.cpu()

            fid_samples_idx = image_labels == 0
            fid_samples = image_samples[fid_samples_idx]

            if replay_buffer.buffer_size is not None:
                replay_buffer.update_buffer(buffer_idx, image_samples, image_labels)

            fid_slice = slice(
                fid_save_sample_idx % time0_bank.shape[0],
                min(
                    time0_bank.shape[0],
                    (fid_save_sample_idx + fid_samples.shape[0]) % time0_bank.shape[0],
                ),
            )
            fid_save_sample_idx += fid_samples.shape[0]
            time0_bank[fid_slice] = fid_samples[: (fid_slice.stop - fid_slice.start)]

            meters["loss"].update(loss.item(), args.batch_size)
            if writer is not None:
                writer.add_scalar("Train/loss_iter", loss.item(), n_iter)

            if idx % args.print_freq == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}] "
                    "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                    "acceptance rate: {acpt_rate:.4f} "
                    "lr {lr:.6} ".format(
                        epoch,
                        idx,
                        len(train_loader) - 1,
                        loss=meters["loss"],
                        acpt_rate=torch.mean(
                            torch.tensor(accumulators["mala_acpt_rate"].average())
                        ),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        else:
            scheduler.step()
            if saved_models_dir is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "step_size": init_step_size.cpu(),
                        "device": device,
                    },
                    is_best=False,
                    save_path_prefix=saved_models_dir,
                )
            if writer is not None:
                writer.add_scalar("Train/Average loss", meters["loss"].avg, epoch)
                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

            if (
                args.start_reject_epochs is not None
                and epoch == args.start_reject_epochs - 1
                and is_reject is False
            ):
                logger.warning("Change Sampler to do proper sampling with rejection")
                is_reject = True

            acpt_rate = accumulators["mala_acpt_rate"].average()
            if (
                args.dynamic_sampling is True
                and is_reject
                and epoch >= args.start_reject_epochs
            ):

                init_step_size = torch.tensor(
                    [
                        adjust_step_size_given_acpt_rate(s_z, a_r)
                        for (s_z, a_r) in zip(
                            init_step_size.cpu(), torch.FloatTensor(acpt_rate),
                        )
                    ]
                ).to(device)
            accumulators["mala_acpt_rate"].reset()

            if writer is not None and is_reject is True:
                for i in range((args.num_diffusion_timesteps + 1)):
                    writer.add_scalar("AcptRate/Time" + str(i), acpt_rate[i], epoch)
                logger.info(
                    "AcptRate "
                    + "".join(
                        [
                            f"t_{i}:{acpt_rate[i]:.2f}; "
                            for i in range((args.num_diffusion_timesteps + 1))
                        ]
                    )
                )
                if args.dynamic_sampling:
                    for i in range(args.num_diffusion_timesteps + 1):
                        writer.add_scalar(
                            "StepSize/Time" + str(i), init_step_size.cpu()[i], epoch
                        )

                    logger.info("step size:" + str(init_step_size.cpu()))

            if (epoch + 1) % args.write_tb_freq == 0:
                labels_jump_mat_freq = accumulators["labels_jump_mat"].average()
                labels_jump_mat_freq = (
                    np.array(labels_jump_mat_freq).reshape(
                        args.num_diffusion_timesteps + 1,
                        args.num_diffusion_timesteps + 1,
                    )
                    / args.batch_size
                )

                rows = [
                    "InitTime %d" % x for x in range(args.num_diffusion_timesteps + 1)
                ]
                columns = [
                    "Time %d" % x for x in range(args.num_diffusion_timesteps + 1)
                ]

                df_cm = pd.DataFrame(
                    labels_jump_mat_freq * 100, index=rows, columns=columns
                )
                fig = plt.figure(figsize=(8, 6))
                ax = sn.heatmap(
                    df_cm, annot=True, fmt=".1f", cbar_kws={"format": "%.0f%%"}
                )
                for t in ax.texts:
                    t.set_text(t.get_text() + " %")
                plt.tight_layout()
                plt.close()
                writer.add_figure(
                    f"TimeStep Jump Table (Average over past {args.write_tb_freq} epochs)",
                    fig,
                    global_step=epoch,
                )
                accumulators["labels_jump_mat"].reset()

                replay_buffer_fig = plot_decision_surface(
                    q, None, time0_bank.squeeze(), None, device, save_path="return"
                )
                writer.add_figure(
                    "Replay Buffer Figure", replay_buffer_fig, global_step=epoch
                )

                fig, axes = plt.subplots(
                    1, args.num_diffusion_timesteps + 1, figsize=(18, 3)
                )
                for i in range(args.num_diffusion_timesteps + 1):
                    image_samples = replay_buffer.buffer_of_samples[
                        replay_buffer.buffer_of_labels == i
                    ]
                    axes[i].scatter(*(image_samples.T))
                plt.tight_layout()
                # plt.show()
                writer.add_figure("Diffusion ReplayBuffer", fig, global_step=epoch)
                plt.close()

                density_along_a_line = plot_diffusion_densities_along_a_line(
                    q, net, device, args.num_diffusion_timesteps + 1
                )

                writer.add_figure(
                    "Density Slice", density_along_a_line, global_step=epoch
                )

            continue
        break

    write_results(args, exp_dir, net, replay_buffer, device)
    # This is needed when handling multiple config files
    logger.removeHandler(fh)
    del logger, fh


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    cli_parser = argparse.ArgumentParser(
        description="configuration arguments provided at run time from the CLI"
    )
    cli_parser.add_argument(
        "-c",
        "--config_files",
        dest="config_files",
        type=str,
        default=[],
        action="append",
        help="config files",
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
        "--main_dir",
        type=str,
        default="./toy",
        help="directory of the experiment: specify as ./toy",
    )

    cli_parser.add_argument(
        "--refresh",
        default=False,
        action="store_true",
        help="whether delete existed exp folder if exists",
    )

    cli_parser.add_argument("--toy_type", type=str, default="rings")
    cli_parser.add_argument("--toy_groups", type=int, default=4)
    cli_parser.add_argument("--toy_sd", type=float, default=1.5e-1)
    cli_parser.add_argument("--toy_radius", type=float, default=1.0)
    cli_parser.add_argument("--viz_res", type=int, default=200)
    cli_parser.add_argument("--kde_bw", type=float, default=0.075)
    cli_parser.add_argument("--b_factor", type=float, default=1e-2)

    args, unknown = cli_parser.parse_known_args()
    add_parser = add_daebm_parser()
    parser = argparse.ArgumentParser(parents=[cli_parser, add_parser], add_help=False)

    args = parser.parse_args()

    main(args)
