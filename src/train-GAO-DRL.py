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
    from lib.config import add_drl_parser
    from lib.diffusion import extract, make_beta_schedule, q_sample
    from lib.train import (configure_net, configure_optimizer,
                           configure_scheduler, get_data_loader,
                           get_dataset_info, initialize_net, save_checkpoint)
    from lib.utils import (AverageMeter, cycle, inv_data_transform,
                           make_figure_grid)
except ImportError:
    raise


def make_sigma_schedule(
    beta_start, beta_end, num_diffusion_timesteps=100, schedule="linear"
):
    """
    Get the noise level schedule
    :param beta_start: begin noise level
    :param beta_end: end noise level
    :param num_diffusion_timesteps: number of timesteps
    :return:
    -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
    -- alphas: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
    """
    betas = make_beta_schedule(
        schedule=schedule, n_timesteps=1000, start=beta_start, end=beta_end
    )
    betas = torch.cat([betas, torch.ones(1)], 0)

    sqrt_alphas = torch.sqrt(1.0 - betas)
    idx = torch.cat(
        [
            torch.arange(num_diffusion_timesteps)
            * (1000 // ((num_diffusion_timesteps - 1) * 2)),
            torch.tensor(999).view(1),
        ],
        0,
    )

    alphas = torch.cat(
        [
            torch.prod(sqrt_alphas[: idx[0] + 1]).view(1),
            torch.tensor(
                [
                    torch.prod(sqrt_alphas[idx[i - 1] + 1 : idx[i] + 1])
                    for i in torch.arange(1, len(idx))
                ]
            ),
        ]
    )
    sigmas = torch.sqrt(1 - alphas ** 2)
    alphas_bar_sqrt = torch.cumprod(alphas, 0)
    alphas_bar_comp_sqrt = torch.sqrt(1 - alphas_bar_sqrt ** 2)

    return sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt


def q_sample_progressive(x_0, alphas, sigmas):
    """
    Generate a full sequence of disturbed images
    """
    x_seq = []
    x_t = x_0
    for t in range(args.num_diffusion_timesteps + 1):
        t_now = torch.ones(x_0.shape[0]).fill_(t).long().to(x_0.device)
        x_t = extract(alphas, t_now, x_t) * x_t + extract(
            sigmas, t_now, x_t
        ) * torch.randn_like(x_0).to(x_0.device)
        x_seq.append(x_t)
    x_seq = torch.stack(x_seq, axis=0)

    return x_seq.cpu()


def q_sample_pairs(x_0, t, sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, alphas_bar_sqrt, alphas_bar_comp_sqrt)
    x_tp1 = extract(alphas, t + 1, x_t) * x_t + extract(sigmas, t + 1, x_t) * noise

    return x_t, x_tp1


def p_sample_langevin(
    net, x_tp1, t, num_steps, sigmas, is_recovery, alphas_prev, init_step_size
):
    """
    Langevin sampling function
    """
    sigmas_tp1 = extract(sigmas, t + 1, x_tp1)
    alphas_tp1 = extract(alphas_prev, t + 1, x_tp1)
    is_recovery_for_x_tp1 = extract(is_recovery, t + 1, x_tp1)

    step_size = extract(init_step_size, t, x_tp1)
    step_size_square = step_size ** 2

    y_t = torch.autograd.Variable(x_tp1.clone(), requires_grad=True)

    def log_prob(net, y_t, t, x_tp1, b0, sigmas_tp1, is_recovery):
        potentials = net.energy_output(y_t, t).squeeze() / b0.squeeze()
        assert len(potentials.shape) == 1
        return potentials + torch.sum(
            (y_t - x_tp1) ** 2 / sigmas_tp1 ** 2 / 2 * is_recovery, axis=(1, 2, 3)
        )

    for _ in torch.arange(num_steps):
        log_p_y = log_prob(
            net, y_t, t, x_tp1, step_size_square, sigmas_tp1, is_recovery_for_x_tp1
        )

        grad_y = torch.autograd.grad(log_p_y.sum(), [y_t], retain_graph=False)[0]
        y_t.data += -0.5 * step_size_square * grad_y + step_size * torch.randn_like(y_t)

    x_t_neg = y_t / alphas_tp1

    return x_t_neg.detach().cpu()


def p_sample_langevin_progressive(
    net,
    noise,
    num_steps,
    sigmas,
    is_recovery,
    alphas_prev,
    init_step_size,
    device="cpu",
):
    """
    Sample a sequence of images with the sequence of noise levels
    """
    num = noise.shape[0]

    x_neg_t = noise
    x_neg = torch.zeros((args.num_diffusion_timesteps, num,) + noise.shape[1:])
    # is_accepted_summary = tf.constant(0.)

    for t in torch.arange(args.num_diffusion_timesteps - 1, -1, -1):
        x_neg_t = p_sample_langevin(
            net,
            x_neg_t.to(device),
            t.repeat(num).long().to(device),
            num_steps,
            sigmas,
            is_recovery,
            alphas_prev,
            init_step_size,
        )
        x_neg[t.long()] = x_neg_t
    x_neg = torch.concat([x_neg, noise.unsqueeze(0).cpu()], axis=0)
    return x_neg


# === Training loss ===
def training_losses(net, x_pos, x_neg, t, alphas_prev, sigmas):
    """
    Training loss calculation
    """
    alphas_tp1 = extract(alphas_prev, t + 1, x_pos)
    y_pos = alphas_tp1 * x_pos
    y_neg = alphas_tp1 * x_neg
    pos_f = net.energy_output(y_pos, t)
    neg_f = net.energy_output(y_neg, t)
    loss = (pos_f - neg_f).squeeze()

    loss_scale = 1.0 / (torch.gather(sigmas, 0, t + 1) / sigmas[1])

    loss = loss_scale * loss

    return loss.mean()


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
    logger = logging.getLogger("DRL Training")
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

    args.n_class = args.num_diffusion_timesteps
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

    net = initialize_net(net.to(device), args.net_init_method)

    optimizer = configure_optimizer(
        net.parameters(),
        args.optimizer_type,
        args.lr,
        args.weight_decay,
        args.betas,
        args.sgd_momentum,
    )

    meter_list = ["loss"]

    sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt = make_sigma_schedule(
        beta_start=args.diffusion_betas[0],
        beta_end=args.diffusion_betas[1],
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        schedule=args.diffusion_schedule,
    )
    alphas_prev = alphas.clone()
    alphas_prev[-1] = 1
    is_recovery = torch.ones(args.num_diffusion_timesteps + 1)
    is_recovery[-1] = 0
    b_square = args.b_factor ** 2

    step_size_square = sigmas ** 2 * b_square
    step_size_square = step_size_square[1:]

    c_t_square = alphas_bar_comp_sqrt / alphas_bar_comp_sqrt[0]
    c_t_square = c_t_square[:(-1)]

    step_size_square *= c_t_square

    init_step_size = step_size_square.sqrt()

    logger.info("sigmas:" + str(sigmas))
    logger.info("alphas:" + str(alphas))
    logger.info("alpha_bar_sqrt:" + str(alphas_bar_sqrt))
    logger.info("alphas_bar_comp_sqrt:" + str(alphas_bar_comp_sqrt))

    logger.info("init_step_size:" + str(init_step_size))

    (
        sigmas,
        alphas,
        alphas_bar_sqrt,
        alphas_bar_comp_sqrt,
        alphas_prev,
        is_recovery,
        init_step_size,
    ) = (
        sigmas.to(device),
        alphas.to(device),
        alphas_bar_sqrt.to(device),
        alphas_bar_comp_sqrt.to(device),
        alphas_prev.to(device),
        is_recovery.to(device),
        init_step_size.to(device),
    )

    image_data_pool = q_sample_progressive(rep_imgs.to(device), alphas, sigmas)
    image_data_pool = inv_data_transform(
        image_data_pool.view((-1,) + args.image_shape), "center_and_scale"
    )
    diffusion_imgs_figure = make_figure_grid(
        image_data_pool,
        nrow=args.num_diffusion_timesteps + 1,
        ncol=10,
        figsize=(8, 0.8 * (args.num_diffusion_timesteps + 1)),
        show=False,
        suptilte=None,
    )
    writer.add_figure("Diffusion Images", diffusion_imgs_figure, global_step=0)

    n_iter = 0
    train_loader = cycle(train_loader)
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}

        for idx in range(iters_per_epoch):
            n_iter += 1

            x_0, _ = train_loader.__next__()
            t = torch.randint(
                high=args.num_diffusion_timesteps, size=(x_0.shape[0],)
            ).to(device)

            x_t, x_tp1 = q_sample_pairs(
                x_0.to(device), t, sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt
            )

            x_t_neg = p_sample_langevin(
                net,
                x_tp1.to(device),
                t.to(device),
                args.sample_steps,
                sigmas,
                is_recovery,
                alphas_prev,
                init_step_size,
            )

            loss = training_losses(net, x_t, x_t_neg.to(device), t, alphas_prev, sigmas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meters["loss"].update(loss.item(), args.batch_size)
            if epoch < args.n_warm_epochs:
                warmup_scheduler.step()

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                logger.error("Training breakdown")
                args.breakdown = "Breakdown"
                break

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
                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

                if (epoch + 1) % args.write_tb_freq == 0:
                    image_samples_pool = p_sample_langevin_progressive(
                        net,
                        torch.randn_like(rep_imgs).to(device),
                        num_steps=args.sample_steps,
                        sigmas=sigmas,
                        is_recovery=is_recovery,
                        alphas_prev=alphas_prev,
                        init_step_size=init_step_size,
                        device=device,
                    )
                    image_samples_pool = inv_data_transform(
                        image_samples_pool.view((-1,) + args.image_shape),
                        "center_and_scale",
                    )

                    recovery_figure = make_figure_grid(
                        image_samples_pool,
                        nrow=args.num_diffusion_timesteps + 1,
                        ncol=10,
                        figsize=(8, (args.num_diffusion_timesteps + 1) * 0.8),
                        show=False,
                        suptilte=None,
                    )
                    writer.add_figure(
                        "Diffusion Samples Figure", recovery_figure, global_step=epoch
                    )

                    image_samples_pool = p_sample_langevin_progressive(
                        net,
                        torch.randn((100,) + args.image_shape).to(device),
                        num_steps=args.sample_steps,
                        sigmas=sigmas,
                        is_recovery=is_recovery,
                        alphas_prev=alphas_prev,
                        init_step_size=init_step_size,
                        device=device,
                    )

                    gen_imgs = inv_data_transform(
                        image_samples_pool[0], "center_and_scale"
                    )

                    gen_imgs_figure = make_figure_grid(
                        gen_imgs, nrow=10, ncol=10, figsize=(10, 10),
                    )

                    writer.add_figure(
                        "Generated Examples", gen_imgs_figure, global_step=epoch
                    )

            save_checkpoint(
                {"epoch": epoch, "state_dict": net.state_dict(), "device": device},
                is_best=True,
                save_path_prefix=saved_models_dir,
            )

            continue
        break

    with open(f"{exp_dir}/net_structure.txt", "w") as f:
        print(net, file=f)
    kwargs = vars(args)
    with open(os.path.join(exp_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=False, indent=4)

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
        required=False,
        default="./mnist",
        help="directory of the experiment: specify as ./mnist, ./fashionmnist",
    )

    cli_parser.add_argument(
        "--refresh",
        default=False,
        action="store_true",
        help="whether delete existed exp folder if exists",
    )

    args, unknown = cli_parser.parse_known_args()
    add_parser = add_drl_parser()
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
