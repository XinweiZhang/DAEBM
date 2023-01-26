import argparse
import json
import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

cwd = os.getcwd()
sys.path.append(cwd)

try:
    from lib.config import add_drl_parser
    from lib.diffusion import extract, make_sigma_schedule, q_sample
    from lib.libtoy import (ToyDataset, plot_decision_surface,
                            plot_diffusion_densities_along_a_line)
    from lib.train import (configure_net, configure_optimizer,
                           configure_scheduler, initialize_net,
                           save_checkpoint)
    from lib.utils import AverageMeter
except ImportError:
    raise


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


def p_sample_langevin(net, x_tp1, t, num_steps, sigmas, alphas, init_step_size):
    """
    Langevin sampling function
    """
    sigmas_tp1 = extract(sigmas, t + 1, x_tp1)
    alphas_tp1 = extract(alphas, t + 1, x_tp1)

    step_size = extract(init_step_size, t, x_tp1)
    step_size_square = step_size ** 2

    y_t = torch.autograd.Variable(x_tp1.clone(), requires_grad=True)

    def log_prob(net, y_t, t, x_tp1, sigmas_tp1):
        potentials = net.energy_output(y_t, t).squeeze()
        assert len(potentials.shape) == 1
        return potentials + torch.sum((y_t - x_tp1) ** 2 / sigmas_tp1 ** 2 / 2, axis=1)

    for _ in torch.arange(num_steps):
        log_p_y = log_prob(net, y_t, t, x_tp1, sigmas_tp1)

        grad_y = torch.autograd.grad(log_p_y.sum(), [y_t], retain_graph=False)[0]
        y_t.data += -0.5 * step_size_square * grad_y + step_size * torch.randn_like(y_t)

    x_t_neg = y_t / alphas_tp1

    return x_t_neg.detach().cpu()


def p_sample_langevin_progressive(
    net, noise, num_steps, sigmas, alphas, init_step_size, device="cpu",
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
            alphas,
            init_step_size,
        )
        x_neg[t.long()] = x_neg_t
    x_neg = torch.concat([x_neg, noise.unsqueeze(0).cpu()], axis=0)
    return x_neg


# === Training loss ===
def training_losses(net, x_pos, x_neg, t, alphas):
    """
    Training loss calculation
    """
    alphas_tp1 = extract(alphas, t + 1, x_pos)
    y_pos = alphas_tp1 * x_pos
    y_neg = alphas_tp1 * x_neg
    pos_f = net.energy_output(y_pos, t)
    neg_f = net.energy_output(y_neg, t)
    loss = (pos_f - neg_f).squeeze()

    return loss.mean()


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

    log_dir = None if args.log_dir == "None" else os.path.join(exp_dir, args.log_dir)

    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("Toy DRL Training")
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

    "Data Loader Configuration"
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

    scheduler, warmup_scheduler = configure_scheduler(
        optimizer,
        args.scheduler_type,
        args.milestones,
        args.lr_decay_factor,
        args.n_epochs,
        n_warm_iters=args.n_warm_epochs * len(train_loader),
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

    b_square = args.b_factor ** 2

    step_size_square = sigmas ** 2 * b_square
    step_size_square = step_size_square[1:]

    init_step_size = step_size_square.sqrt()

    logger.info("sigmas:" + str(sigmas))
    logger.info("alphas:" + str(alphas))
    logger.info("alpha_bar_sqrt:" + str(alphas_bar_sqrt))
    logger.info("alphas_bar_comp_sqrt:" + str(alphas_bar_comp_sqrt))

    logger.info("init_step_size:" + str(init_step_size))

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

    n_iter = 0
    x_0 = next(iter(train_loader))[0]
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}
        for idx, (x_0, _) in enumerate(train_loader):
            n_iter += 1

            t = torch.randint(
                high=args.num_diffusion_timesteps, size=(x_0.shape[0],)
            ).to(device)

            x_t, x_tp1 = q_sample_pairs(
                x_0.to(device), t, sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt
            )

            x_t_neg = p_sample_langevin(
                net,
                x_tp1.to(device),
                t,
                args.sample_steps,
                sigmas,
                alphas,
                init_step_size,
            )

            loss = training_losses(net, x_t, x_t_neg.to(device), t, alphas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                        len(train_loader) - 1,
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
                        torch.randn(5000, 2).to(device),
                        num_steps=args.sample_steps,
                        sigmas=sigmas,
                        alphas=alphas,
                        init_step_size=init_step_size,
                        device=device,
                    )

                    fig, axes = plt.subplots(
                        1, args.num_diffusion_timesteps + 1, figsize=(21, 3)
                    )
                    for i in range(args.num_diffusion_timesteps + 1):
                        image_samples = image_samples_pool[i]
                        axes[i].scatter(*(image_samples.T))
                    plt.tight_layout()
                    # plt.show()
                    writer.add_figure(
                        "Diffusion Samples Figure", fig, global_step=epoch
                    )
                    plt.close()

                    replay_buffer_fig = plot_decision_surface(
                        q, None, image_samples_pool[0], None, device, save_path="return"
                    )
                    writer.add_figure(
                        "Replay Buffer Figure", replay_buffer_fig, global_step=epoch
                    )

                    density_along_a_line = plot_diffusion_densities_along_a_line(
                        q, net, device, args.num_diffusion_timesteps
                    )

                    writer.add_figure(
                        "Density Slice", density_along_a_line, global_step=epoch
                    )
            save_checkpoint(
                {"epoch": epoch, "state_dict": net.state_dict(), "device": device},
                is_best=False,
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
