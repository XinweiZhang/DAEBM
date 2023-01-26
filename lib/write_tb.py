import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch

from lib.diffusion import MSGS_sampling
from lib.train import save_checkpoint
from lib.utils import imshow, inv_data_transform, make_figure_grid


def write_results(args, exp_dir, net, replay_buffer, device):

    kwargs = vars(args)
    with open(os.path.join(exp_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=False, indent=4)

    if replay_buffer.buffer_size is not None:
        save_idx = torch.randperm(replay_buffer.buffer_size)
        if not args.save_all_buffers:
            save_idx = save_idx[0 : min(10000, replay_buffer.buffer_size)]
        replay_buffer_dict = {
            "samples": replay_buffer.buffer_of_samples[save_idx],
            "labels": replay_buffer.buffer_of_labels[save_idx],
        }
        torch.save(replay_buffer_dict, exp_dir + "/replay_buffer.pt")

    with open(f"{exp_dir}/net_structure.txt", "w") as f:
        print(net, file=f)


def ebm_write_tensorboard(
    args,
    *,
    rep_imgs,
    writer,
    net,
    image_samples,
    image_labels,
    sampler,
    replay_buffer,
    meters,
    accumulators,
    epoch,
    logger,
    exp_dir,
    saved_models_dir,
    device,
):

    mala_acpt_rate = torch.mean(torch.tensor(accumulators["mala_acpt_rate"].average()))
    writer.add_scalar("Images/mala_acpt_rate", mala_acpt_rate, epoch)
    if (
        args.start_reject_epochs is not None
        and epoch == args.start_reject_epochs - 1
        and sampler.reject is False
    ):
        logger.warning("Change Sampler to do proper sampling with rejection")
        sampler.reject = True

    accumulators["mala_acpt_rate"].reset()

    if args.save_all_annealed_models is True and (
        (epoch + 1) in args.milestones or (epoch + 1) == args.n_epochs
    ):

        longrun_images, longrun_labels = sampler(
            net,
            rep_imgs,
            None,
            size_each_chain=1,
            burnin=100000,
            slice_window=1,
            sample_labels_freq=args.sample_labels_freq,
        )

        longrun_images = inv_data_transform(longrun_images, args.data_transform)
        longrun_dict = {"init": rep_imgs, "final": longrun_images}
        torch.save(longrun_dict, exp_dir + "/longrun/longrunat" + str(epoch) + ".pt")

        imshow(
            longrun_images,
            ncol=5,
            save_dir=exp_dir + "/longrun/longrunfinalat" + str(epoch) + ".png",
        )

        save_checkpoint(
            {"epoch": epoch, "state_dict": net.state_dict(), "device": device,},
            is_best=False,
            save_path_prefix=saved_models_dir + "_epoch" + str(epoch),
        )

    if epoch % args.write_tb_freq == 0 or ((epoch + 1) == args.n_epochs):

        if replay_buffer.buffer_size is not None:
            replay_buffer_demo_imgs = replay_buffer.buffer_of_samples[
                0 : min(100, args.batch_size)
            ]
        else:
            replay_buffer_demo_imgs = image_samples

        replay_buffer_demo_imgs = inv_data_transform(
            replay_buffer_demo_imgs, args.data_transform
        )
        replay_figure = make_figure_grid(
            replay_buffer_demo_imgs,
            nrow=10,
            ncol=10,
            figsize=(10, int(float(replay_buffer_demo_imgs.shape[0]) / 10)),
        )

        writer.add_figure("Replay Buffer Examples", replay_figure, global_step=epoch)

        # Generate images
        init_samples_print, labels_print, print_idx = replay_buffer.sample_buffer(
            n_samples=10, reinit_probs=args.reinit_probs, deterministic=True,
        )

        images_per_step, labels_per_step = sampler(
            net,
            init_samples_print,
            labels_print,
            size_each_chain=args.sample_steps,
            burnin=0,
        )

        with torch.no_grad():
            energies_per_step = net.energy_output(images_per_step.to(device), None)

        images_per_step = inv_data_transform(images_per_step, args.data_transform)
        figure_digits = make_figure_grid(
            images_per_step,
            None,
            energies_per_step,
            nrow=args.sample_steps,
            ncol=10,
            figsize=(12, args.sample_steps * 1.3),
        )
        writer.add_figure("Digits", figure_digits, global_step=epoch)


def daebm_write_tensorboard(
    args,
    *,
    writer,
    exp_dir,
    saved_models_dir,
    rep_imgs,
    net,
    image_samples,
    image_labels,
    replay_buffer,
    epoch,
    init_step_size,
    is_reject,
    accumulators,
    device,
):

    # Long-run
    if args.save_all_annealed_models is True and (
        epoch == args.milestones[0] - 1 or epoch == args.n_epochs - 1
    ):
        longrun_init_samples = rep_imgs
        longrun_init_labels = torch.zeros(rep_imgs.shape[0]).long()
        longrun_images, longrun_labels, acpt_rate, _ = MSGS_sampling(
            net,
            longrun_init_samples.to(device),
            longrun_init_labels.to(device),
            100,
            init_step_size,
            reject=is_reject,
        )

        longrun_init_samples = inv_data_transform(
            longrun_init_samples, args.data_transform
        )
        longrun_images = inv_data_transform(longrun_images, args.data_transform)
        figure_longrun_init = make_figure_grid(
            longrun_init_samples,
            longrun_init_labels,
            None,
            nrow=10,
            ncol=10,
            figsize=(12, 12),
        )
        figure_longrun_final = make_figure_grid(
            longrun_images.cpu(),
            longrun_labels.cpu(),
            None,
            nrow=10,
            ncol=10,
            figsize=(12, 12),
        )
        longrun_images = inv_data_transform(longrun_images, args.data_transform)
        longrun_dict = {"init": rep_imgs, "final": longrun_images}
        torch.save(longrun_dict, exp_dir + "/longrun/longrunat" + str(epoch) + ".pt")

        imshow(
            longrun_images,
            ncol=5,
            save_dir=exp_dir + "/longrun/longrunfinalat" + str(epoch) + ".png",
        )
        writer.add_figure("Longrun-init", figure_longrun_init, global_step=epoch)
        writer.add_figure("Longrun-final", figure_longrun_final, global_step=epoch)

        save_checkpoint(
            {"epoch": epoch, "state_dict": net.state_dict(), "device": device,},
            is_best=False,
            save_path_prefix=saved_models_dir + "_epoch" + str(epoch),
        )

    if (epoch + 1) % args.write_tb_freq == 0:
        if replay_buffer.buffer_size is not None:
            replay_buffer_demo_lbls = replay_buffer.buffer_of_labels[
                0 : min(100, args.batch_size)
            ]
            replay_buffer_demo_imgs = replay_buffer.buffer_of_samples[
                0 : min(100, args.batch_size)
            ]
        else:
            replay_buffer_demo_lbls = image_labels
            replay_buffer_demo_imgs = image_samples

        replay_buffer_demo_imgs = inv_data_transform(
            replay_buffer_demo_imgs, args.data_transform
        )
        replay_figure = make_figure_grid(
            replay_buffer_demo_imgs,
            replay_buffer_demo_lbls,
            nrow=int(float(replay_buffer_demo_imgs.shape[0]) / 10),
            ncol=10,
            figsize=(10, int(float(replay_buffer_demo_imgs.shape[0]) / 10)),
        )

        writer.add_figure("Replay Buffer Examples", replay_figure, global_step=epoch)

        (
            init_samples_print,
            init_labels_print,
            print_idx,
        ) = replay_buffer.sample_buffer(
            n_samples=20, reinit_probs=args.reinit_probs, deterministic=True,
        )
        _, _, acpt_rate, path_pool = MSGS_sampling(
            net,
            init_samples_print.to(device),
            init_labels_print.to(device),
            args.sample_steps,
            init_step_size=init_step_size,
            reject=is_reject,
            record_path=True,
        )

        images_path = inv_data_transform(
            torch.cat(path_pool["sample"], 0), args.data_transform
        )
        times_path = torch.cat(path_pool["time"], 0)
        energies_path = net.energy_output(images_path.to(device), times_path.to(device))
        figure_digits = make_figure_grid(
            images_path,
            times_path,
            energies_path,
            nrow=args.sample_steps,
            ncol=20,
            figsize=(22, args.sample_steps * 1.3),
            show=False,
        )

        writer.add_figure("Digits", figure_digits, global_step=epoch)

        labels_jump_mat_freq = accumulators["labels_jump_mat"].average()
        torch.save(
            {"jump_mat": labels_jump_mat_freq},
            exp_dir + "/jump_mat/jump_mat_at" + str(epoch) + ".pt",
        )

        labels_jump_mat_freq = (
            np.array(labels_jump_mat_freq).reshape(
                args.num_diffusion_timesteps + 1, args.num_diffusion_timesteps + 1
            )
            / args.batch_size
        )

        rows = ["InitTime %d" % x for x in range(args.num_diffusion_timesteps + 1)]
        columns = ["Time %d" % x for x in range(args.num_diffusion_timesteps + 1)]

        df_cm = pd.DataFrame(labels_jump_mat_freq * 100, index=rows, columns=columns)
        fig = plt.figure(
            figsize=(
                2 + int(0.6 * (args.num_diffusion_timesteps + 1)),
                int(0.6 * (args.num_diffusion_timesteps + 1)),
            )
        )
        ax = sn.heatmap(df_cm, annot=True, fmt=".1f", cbar_kws={"format": "%.0f%%"})
        for _text_ in ax.texts:
            _text_.set_text(_text_.get_text() + " %")
        plt.tight_layout()
        plt.close()
        writer.add_figure(
            f"TimeStep Jump Table (Average over past {args.write_tb_freq} epochs)",
            fig,
            global_step=epoch,
        )
        accumulators["labels_jump_mat"].reset()
