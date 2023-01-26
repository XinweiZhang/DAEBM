import numpy as np
import torch
from torch.nn.functional import softmax


def make_beta_schedule(schedule="linear", n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "sqrtlinear":
        betas = torch.linspace(start, end, n_timesteps) ** 2
    elif schedule == "sqrtcumlinear":
        betas = torch.cumsum(torch.linspace(start, end, n_timesteps), 0) ** 2
    elif schedule == "sqrtlog":
        betas = torch.logspace(-start, -end, n_timesteps) ** 2
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "geometric":
        betas = torch.tensor(
            np.exp(np.linspace(np.log(start), np.log(end), n_timesteps))
        ).float()
    else:
        raise NotImplementedError
    return betas


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
        schedule=schedule,
        n_timesteps=num_diffusion_timesteps,
        start=beta_start,
        end=beta_end,
    )
    betas = torch.cat([torch.zeros(1), betas.clamp_(0, 1)], 0)
    sigmas = betas.sqrt()
    alphas = 1.0 - betas
    alphas_bar_sqrt = torch.cumprod(alphas.sqrt(), 0)
    alphas_bar_comp_sqrt = torch.sqrt(1 - alphas_bar_sqrt ** 2)

    return sigmas, alphas, alphas_bar_sqrt, alphas_bar_comp_sqrt


def extract(input, t, x):
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(x.shape) - 1)
    return out.reshape(*reshape)


def q_sample(x_0, t, alphas_bar_sqrt, alphas_bar_sqrt_comp, noise=None):
    """
    Diffuse the data (t == 0 means diffused 1 step)
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(alphas_bar_sqrt_comp, t, x_0)
    x_t = alphas_t * x_0 + alphas_1_m_t * noise
    return x_t


def MSGS_sampling(
    net, x, t, num_steps, init_step_size, reject=False, record_path=False,
):
    """
    Langevin sampling function
    """
    device = x.device

    acpt_or_reject_list = torch.zeros(num_steps, x.shape[0])
    path_pool = {"sample": [], "time": []}

    step_size = extract(init_step_size, t, x)
    step_size_square = step_size ** 2

    x_t = torch.autograd.Variable(x.clone(), requires_grad=True)
    x_t_prop = torch.autograd.Variable(x.clone(), requires_grad=True)

    U = net.energy_output(x_t, t)
    grad = torch.autograd.grad(U.sum(), [x_t], retain_graph=False)[0]

    for i in torch.arange(num_steps):
        noise = torch.randn_like(x_t)
        x_t_prop.data = x_t.data - 0.5 * step_size_square * grad + step_size * noise

        U_prop = net.energy_output(x_t_prop, t)
        grad_prop = torch.autograd.grad(U_prop.sum(), [x_t_prop], retain_graph=False)[0]
        with torch.no_grad():
            if reject:
                tran_probs = torch.exp(
                    -U_prop
                    + U
                    - 0.5
                    * pow(
                        torch.flatten(
                            -noise + step_size_square / 2 * (grad + grad_prop),
                            start_dim=1,
                        ),
                        2,
                    ).sum(1)
                    + 0.5 * pow(torch.flatten(noise, start_dim=1), 2).sum(1)
                )
                tran_probs = torch.min(
                    tran_probs, torch.ones_like(tran_probs).to(device)
                )
            else:
                tran_probs = torch.ones(x_t.shape[0]).to(device)

        acpt_or_not = torch.rand_like(tran_probs).to(device) < tran_probs

        acpt_or_reject_list[i] = acpt_or_not.cpu()

        U.data[acpt_or_not] = U_prop.data[acpt_or_not]
        x_t.data[acpt_or_not,] = x_t_prop.data[
            acpt_or_not,
        ]
        grad.data[acpt_or_not,] = grad_prop.data[
            acpt_or_not,
        ]

    acpt_rate = torch.tensor(
        [
            acpt_or_reject_list[:, t.cpu() == j].float().nanmean()
            for j in range(len(init_step_size))
        ]
    )

    with torch.no_grad():
        probs = softmax(net.f(x_t), dim=1)

    t = torch.multinomial(probs, 1, replacement=True).squeeze()

    if record_path:
        path_pool["sample"].append(x_t.detach().clone().cpu())
        path_pool["time"].append(t.detach().clone().cpu())

    return x_t.detach().cpu(), t.cpu(), acpt_rate, path_pool


def p_sample_langevin_progressive(
    net, noise, num_steps, init_step_size, reject, inv_temp=1, device="cpu",
):
    """
    Sample a sequence of images with the sequence of noise levels
    """
    num = noise.shape[0]

    x_t_neg = noise
    x_neg = torch.zeros((len(init_step_size), num,) + noise.shape[1:])
    # is_accepted_summary = tf.constant(0.)
    for t in torch.arange(len(init_step_size)).flip(0):
        x_t_neg, t_neg, acpt_rate, _ = MSGS_sampling(
            net,
            x_t_neg.to(device),
            t.repeat(num).long().to(device),
            num_steps,
            init_step_size,
            reject=reject,
        )
        x_neg[t.long()] = x_t_neg

    return x_neg


def q_sample_progressive(x_0, alphas, sigmas):
    """
    Generate a full sequence of disturbed images
    """
    x_seq = []
    x_t = x_0
    for t in range(len(sigmas)):
        t_now = torch.ones(x_0.shape[0]).fill_(t).long().to(x_0.device)
        x_t = extract(alphas, t_now, x_t) * x_t + extract(
            sigmas, t_now, x_t
        ) * torch.randn_like(x_0).to(x_0.device)
        x_seq.append(x_t)
    x_seq = torch.stack(x_seq, axis=0)

    return x_seq.cpu()


def adjust_step_size_given_acpt_rate(step_size, acpt_rate, delta=0.2):

    if torch.isnan(acpt_rate):
        return step_size
    if acpt_rate > 0.8:
        step_size = torch.tensor(1 + 0.5 * delta) * step_size
    elif acpt_rate < 0.6:
        step_size = step_size / (1 + delta)

    return step_size
