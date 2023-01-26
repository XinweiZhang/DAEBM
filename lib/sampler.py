import torch
from torch import exp, flatten, pow

from lib.utils import Accumulator

Normal = torch.distributions.normal.Normal(0, 1)


class ReplayBuffer:
    """
    random_image_type:
        -uniform
        -normal
    """

    def __init__(
        self, buffer_size, image_shape, n_class=None, random_image_type="normal",
    ):
        self.image_shape = image_shape

        if buffer_size is not None:
            self.buffer_of_samples = self.random_images(
                buffer_size, image_shape, type=random_image_type
            )

            self.buffer_of_labels = torch.arange(n_class).repeat_interleave(
                int(buffer_size // n_class) + 1
            )[:buffer_size]

            self.n_class = n_class
            self.buffer_size = buffer_size
        else:
            self.buffer_size = None
        self.random_image_type = random_image_type

    def random_images(self, size, image_shape, type="uniform"):
        """
        type:
            -uniform
            -normal
        """
        if type == "uniform":
            images = torch.zeros((size,) + image_shape).uniform_(0, 1)
        elif type == "normal":
            images = torch.zeros((size,) + image_shape).normal_(0, 1)
        else:
            raise NotImplementedError

        return images

    def sample_buffer(
        self, n_samples, reinit_probs=0.0, deterministic=False,
    ):
        if self.buffer_size is None:
            random_samples = self.random_images(
                n_samples, self.image_shape, type=self.random_image_type
            )
            return random_samples, None

        idx = (
            torch.randint(0, self.buffer_size, (n_samples,))
            if deterministic is False
            else torch.arange(n_samples)
        )

        samples = self.buffer_of_samples[idx]
        labels = self.buffer_of_labels[idx]

        if reinit_probs > 0:
            random_samples = self.random_images(
                n_samples, self.image_shape, type=self.random_image_type
            )
            choose_random = torch.rand(n_samples) < reinit_probs
            samples[choose_random] = random_samples[choose_random]
            labels[choose_random] = (
                torch.zeros(sum(choose_random)).fill_(self.n_class - 1).long()
            )

        return samples, labels, idx

    def update_buffer(self, idx, samples, labels=None):
        self.buffer_of_samples[idx] = samples
        if labels is not None:
            self.buffer_of_labels[idx] = labels


class MALA_Sampling:
    def __init__(
        self,
        reject=True,
        mala_eps=0.5,
        mala_std=None,
        image_shape=(1, 28, 28),
        device="cpu",
    ):
        super().__init__()

        self.image_shape = image_shape
        self.device = device

        if mala_std is None:
            self.mala_eps, self.mala_std = self.pair_mala_std_given_eps(mala_eps)
        else:
            self.mala_eps, self.mala_std = mala_eps, mala_std

        self.reject = reject

        self.last_samples = None
        self.last_labels = None
        self.mala_acpt_rate = None

    def __call__(
        self,
        net,
        init_samples,
        init_labels,
        size_each_chain=1,
        burnin=0,
        slice_window=1,
    ):
        (
            final_samples,
            final_labels,
            self.last_samples,
            self.last_labels,
            self.mala_acpt_rate,
        ) = self.generate_samples(
            net=net,
            init_samples=init_samples,
            init_labels=init_labels,
            reject=self.reject,
            mala_eps=self.mala_eps,
            mala_std=self.mala_std,
            size_each_chain=size_each_chain,
            burnin=burnin,
            slice_window=slice_window,
            device=self.device,
        )

        return final_samples, final_labels

    def reset(self):
        self.last_samples = None
        self.last_labels = None
        self.mala_acpt_rate = None

    def pair_mala_std_given_eps(self, mala_eps):
        mala_eps, mala_std = mala_eps ** 2, mala_eps
        return mala_eps, mala_std

    def cal_energies_and_grads(self, net, samples, labels):

        U = net.energy_output(samples, labels)

        U_grad = torch.autograd.grad(U.sum(), [samples], retain_graph=False)[0]

        return U, U_grad

    def cal_tran_probs(
        self,
        U,
        U_prop,
        U_grad,
        U_prop_grad,
        samples,
        prop_samples,
        noise,
        mala_eps,
        mala_std,
        reject,
        device,
    ):
        if reject:
            with torch.no_grad():
                # ------------without simplification -------------------------------------------------
                # tran_probs = exp(-U_prop + U
                #                  - 1/(2*mala_std**2)*pow(
                #                     flatten(samples - prop_samples
                #                             + mala_eps/2*U_prop_grad, start_dim=1), 2).sum(1)
                #                  + 1/(2*mala_std**2)*pow(
                #                        flatten(prop_samples - samples
                #                                + mala_eps/2*U_grad, start_dim=1), 2).sum(1))
                # ------------------------------------------------------------------------------------

                tran_probs = exp(
                    -U_prop
                    + U
                    - 0.5
                    * pow(
                        flatten(
                            -noise + mala_std / 2 * (U_grad + U_prop_grad), start_dim=1
                        ),
                        2,
                    ).sum(1)
                    + 0.5 * pow(flatten(noise, start_dim=1), 2).sum(1)
                )
                tran_probs = torch.min(
                    tran_probs, torch.ones_like(tran_probs).to(device)
                )

        else:
            tran_probs = torch.ones(samples.shape[0]).to(device)
        return tran_probs

    def generate_samples(
        self,
        net,
        init_samples,
        init_labels,
        reject,
        mala_eps,
        mala_std,
        size_each_chain=1,
        burnin=0,
        slice_window=1,
        device="cpu",
    ):  # noqa

        final_samples_list = []
        final_labels_list = []
        accumulators = {}
        accumulators["mala"] = Accumulator(init_samples.shape[0])

        samples = torch.autograd.Variable(init_samples.clone(), requires_grad=True).to(
            device
        )
        prop_samples = torch.autograd.Variable(
            init_samples.clone(), requires_grad=True
        ).to(device)

        labels = init_labels.to(device) if init_labels is not None else None

        U, U_grad = self.cal_energies_and_grads(net, samples, labels)

        for i in range(size_each_chain * slice_window + burnin):
            noise = torch.randn_like(samples)
            prop_samples.data = samples.data - mala_eps / 2 * U_grad + mala_std * noise

            U_prop, U_prop_grad = self.cal_energies_and_grads(net, prop_samples, labels)

            tran_probs = self.cal_tran_probs(
                U,
                U_prop,
                U_grad,
                U_prop_grad,
                samples,
                prop_samples,
                noise,
                mala_eps,
                mala_std,
                reject,
                device,
            )

            acpt_or_not = torch.rand_like(tran_probs).to(device) < tran_probs
            accumulators["mala"].add(1, acpt_or_not.tolist())

            samples.data[acpt_or_not,] = prop_samples.data[
                acpt_or_not,
            ]
            U.data[acpt_or_not] = U_prop.data[acpt_or_not]
            U_grad.data[acpt_or_not,] = U_prop_grad.data[
                acpt_or_not,
            ]

            if i >= burnin and (i - burnin) % slice_window == 0:
                final_samples_list.append(samples.detach().clone().cpu())
                if init_labels is not None:
                    final_labels_list.append(labels.detach().clone().cpu())

        final_samples = torch.cat(final_samples_list, 0)
        final_labels = (
            torch.cat(final_labels_list, 0) if init_labels is not None else None
        )

        last_samples = samples.detach().cpu()

        last_labels = labels.detach().cpu() if init_labels is not None else None
        mala_acpt_rate = torch.FloatTensor(accumulators["mala"].average())

        return (
            final_samples,
            final_labels,
            last_samples,
            last_labels,
            mala_acpt_rate,
        )
