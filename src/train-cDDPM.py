import argparse
import json
import logging
import os
import re
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
sys.path.append(cwd)

try:
    import lib.sde_sampling as sampling
    from lib.config import add_ddpm_parser
    from lib.diffusion import q_sample
    from lib.sde_lib import VPSDE, get_likelihood_fn, get_score_fn
    from lib.train import (configure_optimizer, configure_scheduler,
                           get_data_loader, get_dataset_info, save_checkpoint)
    from lib.unet import UNet
    from lib.utils import (EMA, AverageMeter, cycle, inv_data_transform,
                           make_figure_grid)
except ImportError:
    raise


def make_bpds_hist(real_energies):
    figure = plt.figure()
    plt.hist(real_energies, density=True)
    plt.legend(loc="upper right")
    return figure


def get_sde_loss_fn(
    sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


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

    log_dir = None if args.log_dir == "None" else os.path.join(exp_dir, args.log_dir)

    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("cDDPM Training")
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
    if "mnist" in args.main_dir:
        args.image_shape = (1, 32, 32)

    torch.manual_seed(args.t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.t_seed)
    np.random.seed(args.t_seed)

    (train_loader, test_loader, rep_imgs,) = get_data_loader(
        args, data_dir, args.batch_size
    )

    sde = VPSDE(
        beta_min=args.diffusion_betas[0],
        beta_max=args.diffusion_betas[1],
        N=args.num_diffusion_timesteps,
    )
    sampling_eps = 1e-3

    # Batch size
    "Networks Configuration"
    if args.model_structure.casefold().startswith("UNet".casefold()):
        net_params = [
            int(_)
            for _ in re.findall(
                r"[-+]?\d*\.\d+|\d+", args.model_structure.replace("UNet", "")
            )
        ]

        net = UNet(
            nf=net_params[0],
            num_res_blocks=net_params[1],
            ch_mult=tuple(net_params[2:]),
            channels=args.image_shape[0],
            resolution=args.image_shape[1],
            act_func=args.model_act_func,
            dropout=args.dp_prob,
            conv_shortcut=args.use_convshortcut,
            resamp_with_conv=args.resamp_with_conv,
        ).to(device)
        net_ema = UNet(
            nf=net_params[0],
            num_res_blocks=net_params[1],
            ch_mult=tuple(net_params[2:]),
            channels=args.image_shape[0],
            resolution=args.image_shape[1],
            act_func=args.model_act_func,
            dropout=args.dp_prob,
            conv_shortcut=args.use_convshortcut,
            resamp_with_conv=args.resamp_with_conv,
        ).to(device)
    else:
        raise NotImplementedError

    logger.info(str(net))

    ema_helper = EMA(0.999)
    ema_helper.register(net)

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

    time0_bank = torch.randn((50000,) + args.image_shape)
    time0_save_sample_idx = 0

    if "mnist" in args.main_dir:
        demo_data = torch.nn.functional.pad(rep_imgs, (2, 2, 2, 2), "constant", -1)
    else:
        demo_data = rep_imgs
    demo_diffusion_levels = [
        0,
        1,
        int(args.num_diffusion_timesteps / 5),
        int(args.num_diffusion_timesteps * 2 / 5),
        int(args.num_diffusion_timesteps * 3 / 5),
        int(args.num_diffusion_timesteps * 4 / 5),
        args.num_diffusion_timesteps - 1,
    ]

    rep_imgs_diffuse = []
    for i in range(len(demo_diffusion_levels)):
        diffusion_level = demo_diffusion_levels[i]
        diffuse_data = q_sample(
            demo_data,
            torch.zeros(demo_data.shape[0]).fill_(diffusion_level).long(),
            sde.sqrt_alphas_cumprod,
            sde.sqrt_alphas_cumprod,
        )
        rep_imgs_diffuse.append(diffuse_data)
    rep_imgs_diffuse = torch.cat(rep_imgs_diffuse, 0)

    figure = make_figure_grid(
        inv_data_transform(rep_imgs_diffuse, args.data_transform),
        nrow=len(demo_diffusion_levels),
        ncol=10,
        figsize=(8, int(len(demo_diffusion_levels) * 0.8)),
        show=False,
    )
    writer.add_figure("Rep. diffusion images", figure, global_step=0)  # plt.show()

    meter_list = ["loss"]

    def get_data_inverse_scaler():
        """Data normalizer. Assume data are always in [0, 1]."""
        return lambda x: (x + 1.0) / 2.0

    inverse_scaler = get_data_inverse_scaler()
    likelihood_fn = get_likelihood_fn(sde, inverse_scaler)

    noise_estimation_loss = get_sde_loss_fn(
        sde, train=True, reduce_mean=True, continuous=True, likelihood_weighting=False
    )

    n_iter = 0

    sampling_shape = (args.batch_size,) + args.image_shape
    sampling_fn = sampling.get_sampling_fn(
        sampler_name="pc",
        predictor="euler_maruyama",
        corrector="none",
        noise_removal=True,
        n_steps_each=1,
        snr=0.16,
        probability_flow=False,
        sde=sde,
        shape=sampling_shape,
        inverse_scaler=inverse_scaler,
        eps=sampling_eps,
        device=device,
    )

    train_loader = cycle(train_loader)
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}
        for idx in range(iters_per_epoch):
            n_iter += 1

            x_0, _ = train_loader.__next__()

            if "mnist" in args.main_dir:
                x_0 = torch.nn.functional.pad(x_0, (2, 2, 2, 2), "constant", -1)

            loss = noise_estimation_loss(net, x_0.to(device),)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_helper.update(net)

            if epoch < args.n_warm_epochs:
                warmup_scheduler.step()

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                logger.error("Training breakdown")
                args.breakdown = "Breakdown"
                break

            meters["loss"].update(loss.item(), args.batch_size)
            if writer is not None:
                writer.add_scalar("Train/loss_iter", loss.item(), n_iter)

            if idx % args.print_freq == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}] "
                    "loss {loss:.4f} "
                    "lr {lr:.6} ".format(
                        epoch,
                        idx,
                        iters_per_epoch - 1,
                        loss=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        else:
            scheduler.step()

            if writer is not None:
                writer.add_scalar("Train/Average loss", meters["loss"].avg, epoch)

            if saved_models_dir is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "EMA": ema_helper.state_dict(),
                        "device": device,
                    },
                    is_best=False,
                    save_path_prefix=saved_models_dir,
                )

            if epoch > 0 and (
                epoch % args.write_tb_freq == 0 or epoch == args.n_epochs - 1
            ):
                ema_helper.ema(net_ema)

                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

                ema_helper.ema(net_ema)

                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

                samples, n = sampling_fn(net)
                time0_samples = samples.cpu()
                fid_slice = slice(
                    time0_save_sample_idx % time0_bank.shape[0],
                    min(
                        time0_bank.shape[0],
                        (time0_save_sample_idx + time0_samples.shape[0])
                        % time0_bank.shape[0],
                    ),
                )
                time0_save_sample_idx += time0_samples.shape[0]
                time0_bank[fid_slice] = time0_samples[: (fid_slice.stop - fid_slice.start)]

                figure = make_figure_grid(
                    time0_samples, nrow=10, ncol=10, figsize=(8, 8), show=False,
                )
                writer.add_figure("Replay Buffer", figure, global_step=epoch)

            continue
        break

    bpds_pool = []
    test_imgs = next(iter(test_loader))[0]
    for test_imgs, _ in iter(test_loader):
        if "mnist" in args.main_dir:
            test_imgs = torch.nn.functional.pad(test_imgs, (2, 2, 2, 2), "constant", -1)
        bpds = likelihood_fn(net_ema, test_imgs.to(device))[0]
        bpds_pool.append(bpds.cpu())
    bpds_pool = torch.cat(bpds_pool, 0)
    figure = make_bpds_hist(bpds_pool.numpy())
    writer.add_figure("Likelihood Histogram", figure, global_step=epoch)

    logger.info("bpds mean: " + str(bpds_pool.mean()))

    torch.save({"bdps": bpds_pool}, f"{exp_dir}/bpds.pt")

    with open(f"{exp_dir}/net_structure.txt", "w") as f:
        print(net, file=f)
    kwargs = vars(args)
    with open(os.path.join(exp_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=False, indent=4)
    # This is needed when handling multiple config files
    logger.removeHandler(fh)
    del logger, fh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
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
        default="./fashionmnist",
        help="directory of the experiment",
    )

    cli_parser.add_argument(
        "--refresh",
        default=False,
        action="store_true",
        help="whether delete existed exp folder if exists",
    )

    args, unknown = cli_parser.parse_known_args()
    add_parser = add_ddpm_parser()
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
