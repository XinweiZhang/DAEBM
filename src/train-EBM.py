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
    from lib.config import add_ebm_parser
    from lib.sampler import MALA_Sampling, ReplayBuffer
    from lib.train import (configure_net, configure_optimizer,
                           configure_scheduler, get_data_loader,
                           get_dataset_info, initialize_net, save_checkpoint)
    from lib.utils import Accumulator, AverageMeter, cycle
    from lib.write_tb import ebm_write_tensorboard, write_results
except ImportError:
    raise


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def training_losses(net, x, x_s, device):

    x, x_s = x.to(device), x_s.to(device)

    energies = net.energy_output(x)
    energies_s = net.energy_output(x_s)

    loss = energies.mean() - energies_s.mean()

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

    log_dir = None if args.log_dir == "None" else os.path.join(exp_dir, args.log_dir)

    logging.basicConfig(
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("EBM Training")
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
    args.n_class = 1
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
    sampler = MALA_Sampling(
        reject=is_reject,
        mala_eps=args.mala_eps,
        mala_std=args.mala_std,
        image_shape=args.image_shape,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        buffer_size=args.replay_buffer_size,
        image_shape=args.image_shape,
        n_class=1,
        random_image_type=args.random_image_type,
    )

    n_iter = 0
    break_down_count = 0

    accumulators = {}
    accumulators["mala_acpt_rate"] = Accumulator(args.batch_size)

    meter_list = ["loss"]
    train_loader = cycle(train_loader)
    for epoch in range(args.n_epochs):
        meters = {key: AverageMeter() for key in meter_list}

        for idx in range(iters_per_epoch):
            n_iter += 1

            x, _ = train_loader.__next__()

            init_samples, init_labels, buffer_idx = replay_buffer.sample_buffer(
                n_samples=args.batch_size, reinit_probs=args.reinit_probs,
            )

            image_samples, image_labels = sampler(
                net=net,
                init_samples=init_samples,
                init_labels=init_labels,
                size_each_chain=1,
                burnin=args.sample_steps,
            )

            accumulators["mala_acpt_rate"].add(1, sampler.mala_acpt_rate)

            loss = training_losses(net, x, image_samples, device)

            if torch.isnan(loss) or loss.abs().item() > 1e8:
                if break_down_count < 3:
                    checkpoint = torch.load(
                        exp_dir + "/saved_models/net_checkpoint.pt",
                        map_location=device,
                    )
                    net.load_state_dict(checkpoint["state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    break_down_count += 1
                    logger.error(f"Training breakdown: {break_down_count}th time")
                    continue
                else:
                    logger.error("Training breakdown")
                    args.breakdown = "Breakdown"
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < args.n_warm_epochs:
                warmup_scheduler.step()

            if replay_buffer.buffer_size is not None:
                replay_buffer.update_buffer(
                    buffer_idx,
                    sampler.last_samples,
                    image_labels if image_labels is not None else None,
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
                        args.iters_per_epoch - 1,
                        loss=meters["loss"],
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
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
        else:  # only executed if the inner loop did NOT break
            scheduler.step()
            if saved_models_dir is not None:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "device": device,
                    },
                    is_best=False,
                    save_path_prefix=saved_models_dir,
                )

            if writer is not None:
                writer.add_scalar("Train/Average loss", meters["loss"].avg, epoch)
                writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

                ebm_write_tensorboard(
                    args,
                    rep_imgs=rep_imgs,
                    writer=writer,
                    net=net,
                    image_samples=image_samples,
                    image_labels=image_labels,
                    sampler=sampler,
                    replay_buffer=replay_buffer,
                    meters=meters,
                    accumulators=accumulators,
                    epoch=epoch,
                    logger=logger,
                    exp_dir=exp_dir,
                    saved_models_dir=saved_models_dir,
                    device=device,
                )
            continue
        break

    write_results(args, exp_dir, net, replay_buffer, device)

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
    add_parser = add_ebm_parser()
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
