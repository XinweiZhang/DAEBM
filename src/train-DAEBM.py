import argparse
import json
import logging
import os
import shutil
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.config import add_daebm_parser
    from lib.diffusion import (MSGS_sampling, adjust_step_size_given_acpt_rate,
                               make_sigma_schedule, q_sample)
    from lib.sampler import ReplayBuffer
    from lib.train import (configure_net, configure_optimizer,
                           configure_scheduler, get_data_loader,
                           get_dataset_info, initialize_net, save_checkpoint)
    from lib.utils import (Accumulator, AverageMeter, cycle, imshow,
                           inv_data_transform, make_figure_grid)
    from lib.write_tb import daebm_write_tensorboard, write_results
except ImportError:
    raise

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


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
    data_dir = os.path.join(args.main_dir, args.data_dir)

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
    logger = logging.getLogger("DA-EBM Training")
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

    "Data Loader Configuration"
    dataset_info = get_dataset_info(args.main_dir.replace("./", ""))
    args.image_shape = dataset_info["image_shape"]

    torch.manual_seed(args.t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.t_seed)
    np.random.seed(args.t_seed)

    (train_loader, test_loader, rep_imgs,) = get_data_loader(
        args, data_dir, args.batch_size
    )

    if args.save_all_annealed_models:
        os.makedirs(exp_dir + "/longrun")
        imshow(rep_imgs, ncol=5, save_dir=exp_dir + "/longrun/longruninit.png")
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
    iters_per_epoch = args.iters_per_epoch
    scheduler, warmup_scheduler = configure_scheduler(
        optimizer,
        args.scheduler_type,
        args.milestones,
        args.lr_decay_factor,
        args.n_epochs,
        n_warm_iters=args.n_warm_epochs * iters_per_epoch,
    )

    "Sampling Configuration"
    is_reject = args.mala_reject
    replay_buffer = ReplayBuffer(
        buffer_size=args.replay_buffer_size,
        image_shape=args.image_shape,
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
    step_size_square = torch.ones_like(sigmas) * args.init_step_size_values

    init_step_size = step_size_square.sqrt()

    logger.info("sigmas:" + str(sigmas))
    logger.info("alphas:" + str(alphas))
    logger.info("alpha_bar_sqrt:" + str(alphas_bar_sqrt))
    logger.info("alphas_bar_comp_sqrt:" + str(alphas_bar_comp_sqrt))

    logger.info("step_size:" + str(init_step_size))

    rep_imgs_denoise = torch.cat(
        [
            q_sample(
                rep_imgs,
                torch.zeros(rep_imgs.shape[0]).fill_(t).long(),
                alphas_bar_sqrt,
                alphas_bar_comp_sqrt,
            )
            for t in torch.arange(args.num_diffusion_timesteps + 1)
        ],
        0,
    )

    figure = make_figure_grid(
        inv_data_transform(rep_imgs_denoise, args.data_transform),
        nrow=args.num_diffusion_timesteps + 1,
        ncol=10,
        figsize=(8, int((args.num_diffusion_timesteps + 1) * 0.8)),
        show=False,
    )
    writer.add_figure("Rep. diffusion images", figure, global_step=0)

    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt, init_step_size = (
        sigmas.to(device),
        alphas.to(device),
        alphas_bar_sqrt.to(device),
        alphas_bar_comp_sqrt.to(device),
        init_step_size.to(device),
    )

    "Ancillary"
    n_iter = 0
    time0_bank = torch.randn((50000,) + args.image_shape)  # Store Time 0 images
    fid_save_sample_idx = 0

    accumulators = {}
    accumulators["mala_acpt_rate"] = Accumulator(args.num_diffusion_timesteps + 1)
    accumulators["labels_jump_mat"] = Accumulator(
        (args.num_diffusion_timesteps + 1) ** 2
    )

    meter_list = ["loss"]

    train_loader = cycle(train_loader)
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}

        for idx in range(iters_per_epoch):
            n_iter += 1

            x_0, _ = train_loader.__next__()

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
            time0_bank[fid_slice] = inv_data_transform(
                fid_samples[: (fid_slice.stop - fid_slice.start)], args.data_transform
            )

            meters["loss"].update(loss.item(), args.batch_size)
            if writer is not None:
                writer.add_scalar("Train/loss_iter", loss.item(), n_iter)

            if idx % args.print_freq == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}] "
                    "Loss {loss.val:.4f} ({loss.avg:.4f}) "
                    "lr {lr:.5f} ".format(
                        epoch,
                        idx,
                        iters_per_epoch - 1,
                        loss=meters["loss"],
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
                if len(image_samples) != 0:
                    logger.info(
                        "acceptance rate: {acpt_rate:.4f} "
                        "pixels mean: {pixels_mean:.4f} "
                        "pixels min: {pixels_min:.4f} "
                        "pixels max: {pixels_max:.4f} ".format(
                            acpt_rate=torch.mean(
                                torch.tensor(accumulators["mala_acpt_rate"].average())
                            ),
                            pixels_mean=image_samples.mean().item(),
                            pixels_min=image_samples.min().item(),
                            pixels_max=image_samples.max().item(),
                        )
                    )

            if n_iter % args.adjust_step_size_freq == 0:
                acpt_rate = accumulators["mala_acpt_rate"].average()
                if (
                    args.dynamic_sampling is True
                    and is_reject
                    and epoch >= args.start_reject_epochs
                ):

                    adjust_step_size = torch.tensor(
                        [
                            adjust_step_size_given_acpt_rate(s_z, a_r)
                            for (s_z, a_r) in zip(
                                init_step_size.cpu(), torch.FloatTensor(acpt_rate),
                            )
                        ]
                    )
                    if torch.any(torch.ne(adjust_step_size, init_step_size.cpu())):
                        logger.warning("adjust step size:" + str(adjust_step_size))
                        init_step_size = adjust_step_size.to(device)

                accumulators["mala_acpt_rate"].reset()

        else:  # only executed if the inner loop did NOT break
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

                daebm_write_tensorboard(
                    args,
                    writer=writer,
                    exp_dir=exp_dir,
                    saved_models_dir=saved_models_dir,
                    rep_imgs=rep_imgs,
                    net=net,
                    image_samples=x_t_neg,
                    image_labels=t_neg,
                    replay_buffer=replay_buffer,
                    epoch=epoch,
                    init_step_size=init_step_size,
                    is_reject=is_reject,
                    accumulators=accumulators,
                    device=device,
                )
            if (
                args.start_reject_epochs is not None
                and epoch == args.start_reject_epochs - 1
                and is_reject is False
            ):
                logger.warning("Change Sampler to do proper sampling with rejection")
                is_reject = True

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

            continue
        break

    write_results(args, exp_dir, net, replay_buffer, device)

    writer.close()
    logger.removeHandler(fh)  # This is needed when handling multiple config files
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
        required=True,
        help="directory of the experiment: specify as ./mnist, ./fashionmnist",
    )

    cli_parser.add_argument(
        "--refresh",
        default=False,
        action="store_true",
        help="whether delete existed exp folder if exists",
    )

    args, unknown = cli_parser.parse_known_args()
    add_parser = add_daebm_parser()
    parser = argparse.ArgumentParser(parents=[cli_parser, add_parser], add_help=False)

    if args.help_more:
        parser.print_help()
    if args.config_files:
        for config_file in args.config_files:
            with open(os.path.join(args.main_dir, f"configs/{config_file}"), "r") as fp:
                config = json.load(fp)

            parser.set_defaults(**config)

            args = parser.parse_args()

            main(args)
    else:
        args = parser.parse_args()

        main(args)
