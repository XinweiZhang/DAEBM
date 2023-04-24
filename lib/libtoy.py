import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, multivariate_normal, truncnorm


class ToyDataset:
    def __init__(
        self,
        toy_type="gmm",
        toy_groups=8,
        toy_sd=0.15,
        toy_radius=1,
        viz_res=500,
        kde_bw=0.05,
    ):
        # import helper functions
        self.gaussian_kde = gaussian_kde
        self.mvn = multivariate_normal
        self.truncnorm = truncnorm

        # toy dataset parameters
        self.toy_type = toy_type
        self.toy_groups = toy_groups
        self.toy_sd = toy_sd
        self.toy_radius = toy_radius
        self.weights = np.ones(toy_groups) / toy_groups
        if toy_type == "gmm":
            means_x = np.cos(
                2 * np.pi * np.linspace(0, (toy_groups - 1) / toy_groups, toy_groups)
            ).reshape(toy_groups, 1)
            means_y = np.sin(
                2 * np.pi * np.linspace(0, (toy_groups - 1) / toy_groups, toy_groups)
            ).reshape(toy_groups, 1)
            self.means = toy_radius * np.concatenate((means_x, means_y), axis=1)
        else:
            self.means = None

        # ground truth density
        if self.toy_type == "gmm":

            def true_density(x):
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k] * self.mvn.pdf(
                        np.array([x[1], x[0]]),
                        mean=self.means[k].squeeze(),
                        cov=(self.toy_sd ** 2) * np.eye(2),
                    )
                return density

        elif self.toy_type == "rings":

            def true_density(x):
                radius = np.sqrt((x[1] ** 2) + (x[0] ** 2))
                density = 0
                for k in range(toy_groups):
                    # density += self.weights[k] * self.mvn.pdf(radius, mean=self.toy_radius * (k + 1),
                    #                                           cov=(self.toy_sd**2))/(2*np.pi*self.toy_radius*(k+1))
                    density += (
                        self.weights[k]
                        * self.truncnorm.pdf(
                            radius,
                            a=(0 - self.toy_radius * (k + 1)) / self.toy_sd,
                            b=np.inf,
                            loc=self.toy_radius * (k + 1),
                            scale=self.toy_sd,
                        )
                        / (2 * np.pi)
                    )

                return density

        else:
            raise RuntimeError('Invalid option for toy_type (use "gmm" or "rings")')
        self.true_density = true_density

        # viz parameters
        self.viz_res = viz_res
        self.kde_bw = kde_bw
        if toy_type == "rings":
            self.plot_val_max = toy_groups * toy_radius + 4 * toy_sd
        else:
            self.plot_val_max = toy_radius + 4 * toy_sd

        # save values for plotting groundtruth landscape
        self.xy_plot = np.linspace(-self.plot_val_max, self.plot_val_max, self.viz_res)
        self.z_true_density = np.zeros(self.viz_res ** 2).reshape(
            self.viz_res, self.viz_res
        )
        for x_ind in range(len(self.xy_plot)):
            for y_ind in range(len(self.xy_plot)):
                self.z_true_density[x_ind, y_ind] = self.true_density(
                    [self.xy_plot[x_ind], self.xy_plot[y_ind]]
                )

    def sample_toy_data(self, num_samples):
        toy_sample = np.zeros(0).reshape(0, 2)
        sample_group_sz = np.random.multinomial(num_samples, self.weights)
        if self.toy_type == "gmm":
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd * np.random.randn(
                    2 * sample_group_sz[i]
                ).reshape(-1, 2)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
        elif self.toy_type == "rings":
            for i in range(self.toy_groups):
                truncnorm_rv = self.truncnorm(
                    a=(0 - self.toy_radius * (i + 1)) / self.toy_sd,
                    b=np.inf,
                    loc=self.toy_radius * (i + 1),
                    scale=self.toy_sd,
                )
                # sample_radii = self.toy_radius*(i+1) + self.toy_sd * np.random.randn(sample_group_sz[i])
                sample_radii = truncnorm_rv.rvs(sample_group_sz[i])
                sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
                sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(
                    -1, 1
                )
                sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(
                    -1, 1
                )
                sample_group = np.concatenate((sample_x, sample_y), axis=1)
                toy_sample = np.concatenate(
                    (toy_sample, sample_group.reshape(-1, 2)), axis=0
                )
        else:
            raise RuntimeError('Invalid option for toy_type ("gmm" or "rings")')

        toy_label = np.concatenate(
            [np.array(i).repeat(sample_group_sz[i]) for i in range(len(self.weights))]
        )

        return toy_sample, toy_label


def plot_decision_surface(q, net, points, labels, device, save_path=None):
    xx, yy = np.meshgrid(
        np.arange(-q.plot_val_max, q.plot_val_max, 0.1),
        np.arange(-q.plot_val_max, q.plot_val_max, 0.1),
    )

    xxyy = np.concatenate(
        (np.ones((xx.shape[0] * xx.shape[1], 1)), np.c_[xx.ravel(), yy.ravel()]), axis=1
    )
    xxyy = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()].reshape(-1, 2), dtype=torch.float32
    )
    if net is not None:
        pred = net(xxyy.to(device))

    fig, ax = plt.subplots()
    title = "Decision surface"
    if net is not None:
        ax.contourf(
            xx,
            yy,
            pred.argmax(1).cpu().numpy().reshape(xx.shape[0], -1),
            alpha=0.75,
            cmap=plt.cm.coolwarm,
        )
    ax.scatter(
        points[:, 0], points[:, 1], c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors="k"
    )

    ax.set_title(title)
    if save_path is None:
        plt.show()
    elif save_path == "return":
        return fig
    else:
        plt.savefig(save_path, bbox_inches="tight", format="pdf")
        plt.close()


def plot_density_along_a_line(
    q, net, device, figsize=(10, 2.5), rescale=None, n_class=1, plot_range=None
):
    if plot_range is None:
        plot_range = 1.5 * q.plot_val_max
    yy = torch.arange(-plot_range, plot_range, 0.01).unsqueeze(1)
    data = torch.cat([torch.zeros_like(yy), yy], 1).view(-1, 2)

    potentials = net.energy_output(data.to(device), None).detach().cpu()
    if rescale is not None:
        potentials *= 2 / (rescale ** 2)

    densities_unnormalized = torch.exp(-potentials)
    # bin_area = (yy[0] - yy[1])**2
    densities = torch.exp(-(potentials - potentials.min()))
    potentials_normalized = potentials - potentials.min()
    # densities = densities / (bin_area * sum(densities))
    true_densities = torch.tensor(
        [q.true_density(data[i].squeeze()) for i in torch.arange(len(data))]
    )
    fig, ax = plt.subplots(1, 4, figsize=figsize)

    ax[0].plot(yy, true_densities.numpy(), linewidth=2, markersize=12, label="True")
    ax[0].plot(
        yy, densities_unnormalized.numpy(), linewidth=2, markersize=12, label="Learned"
    )
    ax[0].set_title("Unnormalized Density")
    ax[1].plot(yy, true_densities.numpy(), linewidth=2, markersize=12, label="True")
    ax[1].plot(yy, densities.numpy(), linewidth=2, markersize=12, label="Learned")
    ax[1].set_title("Normalized Density")
    ax[2].plot(
        yy, -true_densities.log().numpy(), linewidth=2, markersize=12, label="True"
    )
    ax[2].plot(yy, potentials.numpy(), linewidth=2, markersize=12, label="Learned")
    ax[2].set_title("Unnormalized Potential")
    ax[3].plot(
        yy,
        -true_densities.log().numpy() - (-true_densities.log().numpy()).min(),
        linewidth=2,
        markersize=12,
        label="True",
    )
    ax[3].plot(
        yy, potentials_normalized.numpy(), linewidth=2, markersize=12, label="Learned"
    )
    ax[3].set_ylim(
        [
            -potentials_normalized.numpy().max() * 0.05,
            potentials_normalized.numpy().max() * 1.05,
        ]
    )
    ax[3].set_title("Normalized Potential")

    handles, labels = ax[2].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.15, 0.1))
    plt.close()

    return fig


def plot_diffusion_potentials_along_a_line(q, net, device, num_diffusion_timesteps=1):

    fig, ax = plt.subplots(num_diffusion_timesteps, 1, figsize=(50, 4*num_diffusion_timesteps))
    for idx in range(num_diffusion_timesteps):
        if isinstance(q, list):
            yy = torch.arange(0.005, 1.2*q[idx].plot_val_max, 0.01).unsqueeze(1)
            yy = torch.cat([-yy.flip(0), yy], 0)
        else:
            yy = torch.arange(0.005, 1.2*q.plot_val_max, 0.01).unsqueeze(1)
            yy = torch.cat([-yy.flip(0), yy], 0)

        data = torch.cat([torch.zeros_like(yy), yy], 1).view(-1, 2)

        learned_potentials = net.energy_output(
            data.to(device), torch.zeros(data.shape[0]).fill_(idx).long().to(device)).detach().cpu()

        learned_potentials_normalized = learned_potentials - learned_potentials.min()

        # bin_area = (yy[0] - yy[1])**2
        # densities = densities / (bin_area * sum(densities))
        if isinstance(q, list):
            true_densities = torch.tensor([q[idx].true_density(data[i].squeeze()) for i in torch.arange(len(data))])
        else:
            true_densities = torch.tensor([q.true_density(data[i].squeeze()) for i in torch.arange(len(data))])
        true_potential = -true_densities.log()
        true_potential_normalized = true_potential - true_potential.min()

        ax[idx].plot(yy, true_potential_normalized, linewidth=2, markersize=12, label="True")
        ax[idx].plot(yy, learned_potentials_normalized.numpy(), linewidth=2, markersize=12, label="Learned")
        ax[idx].set_ylim([-learned_potentials_normalized.numpy().max()*0.05, learned_potentials_normalized.numpy().max()*1.05])
        ax[idx].set_title("Normalized Potential Time " + str(idx))

        handles, labels = ax[idx].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.15, 0.1))
    plt.tight_layout()
    # plt.show()
    plt.close()

    return fig


def plot_diffusion_densities_along_a_line(q, net, device, num_diffusion_timesteps=1):

    fig, ax = plt.subplots(
        num_diffusion_timesteps, 4, figsize=(10, 2.5 * num_diffusion_timesteps)
    )
    for idx in range(num_diffusion_timesteps):
        if isinstance(q, list):
            yy = torch.arange(0.005, 1.2 * q[idx].plot_val_max, 0.01).unsqueeze(1)
            yy = torch.cat([-yy.flip(0), yy], 0)
        else:
            yy = torch.arange(0.005, 1.2 * q.plot_val_max, 0.01).unsqueeze(1)
            yy = torch.cat([-yy.flip(0), yy], 0)

        data = torch.cat([torch.zeros_like(yy), yy], 1).view(-1, 2)

        learned_potentials = (
            net.energy_output(
                data.to(device), torch.zeros(data.shape[0]).fill_(idx).long().to(device)
            )
            .detach()
            .cpu()
        )

        learned_densities = torch.exp(-learned_potentials)
        learned_potentials_normalized = learned_potentials - learned_potentials.min()
        learned_densities_normalized = torch.exp(-learned_potentials_normalized)

        # bin_area = (yy[0] - yy[1])**2
        # densities = densities / (bin_area * sum(densities))
        if isinstance(q, list):
            true_densities = torch.tensor(
                [
                    q[idx].true_density(data[i].squeeze())
                    for i in torch.arange(len(data))
                ]
            )
        else:
            true_densities = torch.tensor(
                [q.true_density(data[i].squeeze()) for i in torch.arange(len(data))]
            )
        true_potential = -true_densities.log()
        true_potential_normalized = true_potential - true_potential.min()
        true_densities_normalized = torch.exp(-true_potential_normalized)

        ax[idx][0].plot(
            yy,
            true_densities.numpy(),
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )
        ax[idx][0].plot(
            yy, learned_densities.numpy(), linewidth=2, markersize=12, label="Learned"
        )
        ax[idx][0].set_title("Unnormalized Density Time " + str(idx))
        ax[idx][1].plot(
            yy,
            true_densities_normalized.numpy(),
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )
        ax[idx][1].plot(
            yy,
            learned_densities_normalized.numpy(),
            linewidth=2,
            markersize=12,
            label="Learned",
        )
        ax[idx][1].set_title("Normalized Density Time " + str(idx))
        ax[idx][2].plot(
            yy,
            true_potential.numpy(),
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )
        ax[idx][2].plot(
            yy, learned_potentials.numpy(), linewidth=2, markersize=12, label="Learned"
        )
        ax[idx][2].set_ylim(
            [
                learned_potentials.numpy().min() - 0.5,
                learned_potentials.numpy().max() + 0.5,
            ]
        )
        ax[idx][2].set_title("Unnormalized Potential Time " + str(idx))
        ax[idx][3].plot(
            yy,
            true_potential_normalized,
            linewidth=2,
            markersize=12,
            label="True",
            color="#ff7f0e",
        )
        ax[idx][3].plot(
            yy,
            learned_potentials_normalized.numpy(),
            linewidth=2,
            markersize=12,
            label="Learned",
        )
        ax[idx][3].set_ylim(
            [
                -learned_potentials_normalized.numpy().max() * 0.05,
                learned_potentials_normalized.numpy().max() * 1.05,
            ]
        )
        ax[idx][3].set_title("Normalized Potential Time " + str(idx))

        handles, labels = ax[idx][3].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.15, 0.1))
    plt.tight_layout()
    # plt.show()
    plt.close()

    return fig
