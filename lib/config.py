import argparse


def check_none_or_float(x):
    if x == "None":
        return None
    else:
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%r not a floating-point literal" % (x,))
        return x


def check_none_or_int(x):
    if x == "None":
        return None
    else:
        try:
            x = int(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a integer literal" % (x,))
        return x


def add_ebm_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp_dir", type=str, default="experiments/",
                        help="directory of the experiment")

    parser.add_argument("--data_dir", type=str,
                        default="data/", help="path to dataset")

    parser.add_argument("--t_seed", type=int, default=1234, help="manual seed")

    parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")

    parser.add_argument("--cuda", type=int, default="0", help="device-name")

    parser.add_argument('--save_all_annealed_models', action="store_true", default=False,
                        help='save all models when anneled learning rate')

    "Log Parameters"
    parser.add_argument("--log_dir", type=str, default="tb_log/",
                        help="path and prefix of tensorboard log file")

    parser.add_argument("--print_freq", type=int, default=100,
                        help="number of iterations for printing log during training")

    parser.add_argument('--write_tb_freq', type=int,
                        default=25, help='write tensorboard frequency')

    "Learning Parameters"
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs to training")

    parser.add_argument("--n_warm_epochs", type=int,
                        default=10, help="number of epochs to training")

    parser.add_argument("--iters_per_epoch", type=int,
                        default=600, help="number of epochs to training")

    parser.add_argument("--batch_size", type=int, default=200,
                        help="batch size of the validation and testing data used during training")

    "Optimizer Parameters"
    parser.add_argument("--optimizer_type", dest="optimizer_type",
                        type=str, default="adam", help="adam or sgd")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument('--betas', nargs=2, metavar=('beta1', 'beta2'), default=(0.9, 0.999),
                        type=float, help="beta parameters for adam optimizer")

    parser.add_argument('--sgd_momentum', type=float,
                        default=0.0, help='momentum in sgd')

    parser.add_argument("--weight_decay", type=float,
                        default=.0, help="weight decay for discriminator")

    parser.add_argument('--scheduler_type', type=str, default="LinearPlateau",
                        help='learning rate scheduler: MultiStep, Linear, LinearPlateau')

    parser.add_argument('--lr_decay_factor', type=float, default=1e-4,
                        help='learning rate decay factor')

    parser.add_argument('--milestones', nargs="+", type=int, default=[120, 180],
                        help='milestones of learning rate decay')

    "Network Parameters"
    parser.add_argument('--model_structure', type=str,
                        default="mlp", help='model_structure',)

    parser.add_argument('--model_act_func', type=str,
                        default="lReLU[0.1]", help='model activation function',)

    parser.add_argument("--net_init_method", dest="net_init_method", type=str, default="kaiming",
                        help="network initialization method")

    parser.add_argument('--resamp_with_conv', action="store_true", default=False,
                        help='do down sampling with conv net in the network')

    parser.add_argument('--use_convshortcut', action="store_true", default=False,
                        help='use conv shortcut in the network')

    parser.add_argument('--dp_prob', type=float, default=0,
                        help='dropout probability')

    parser.add_argument("--network_normalization", dest="network_normalization", type=str, default=None,
                        help="network normalization")

    parser.add_argument(
        "--use_spectral_norm",
        default=False,
        action="store_true",
        help="whether use spectral norm",
    )

    "Data Preprocessing"
    parser.add_argument(
        "--uniform_dequantization",
        default=False,
        action="store_true",
        help="whether use uniform dequantization",
    )

    parser.add_argument('--gaussian_noise_std', type=float, default=None,
                        help='standard deviation of guassion noise in data augmentation')

    parser.add_argument('--random_horizontal_flip', action="store_true",
                        default=False,  help='random horizontal flip image')

    parser.add_argument("--data_augmentation", dest="data_augmentation", action="store_true",
                        help="do data augmentation")
    parser.add_argument("--no-data_augmentation",
                        dest="data_augmentation", action="store_false")
    parser.set_defaults(data_augmentation=True)

    parser.add_argument('--data_transform', type=str,
                        default="center_and_scale", help='transformation on the data',)

    "Sampling Parameters"
    parser.add_argument("--mala_reject",  dest="mala_reject", action="store_true",
                        default="no-reject", help="MALA sampler, rejection or no rejection")
    parser.add_argument("--no-mala_reject", dest="mala_reject", action="store_false")
    parser.set_defaults(mala_reject=False)

    parser.add_argument("--random_image_type", type=str, default="normal",
                        help="random images type, uniform, norma")

    parser.add_argument("--sampling_dist", type=str,
                        default="marginal", help="sampling distribution")

    parser.add_argument("--mala_eps", type=float, default=2,
                        help="step-size of MALA sampler")

    parser.add_argument("--mala_std", type=check_none_or_float, default=0.01,
                        help="noise level of MALA sampler")

    parser.add_argument("--sample_steps", type=int,
                        default=40, help="sampling steps")

    parser.add_argument('--replay_buffer_size', type=check_none_or_int, default=None,
                        help='replay buffer size, if not speicified, we run short-run')

    parser.add_argument('--reinit_probs', type=float, default=0.0,
                        help='replay buffer, reinitialize probability')

    parser.add_argument('--start_reject_epochs', type=int,
                        default=None, help='start_reject_epochs')

    parser.add_argument('--save_all_buffers', action="store_true", default=False,
                        help='save all buffer')

    return parser


def add_daebm_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp_dir", type=str, default="experiments/",
                        help="directory of the experiment")

    parser.add_argument("--data_dir", type=str,
                        default="data/", help="path to dataset")

    parser.add_argument("--t_seed", type=int, default=1234, help="manual seed")

    parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")

    parser.add_argument("--cuda", type=int, default="0", help="device-name")

    parser.add_argument('--save_all_annealed_models', action="store_true", default=False,
                        help='save all models when anneled learning rate')

    "Log Parameters"
    parser.add_argument("--log_dir", type=str, default="tb_log/",
                        help="path and prefix of tensorboard log file")

    parser.add_argument("--print_freq", type=int, default=100,
                        help="number of iterations for printing log during training")

    parser.add_argument('--write_tb_freq', type=int,
                        default=25, help='write tensorboard frequency')

    "Learning Parameters"
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs to training")

    parser.add_argument("--n_warm_epochs", type=int,
                        default=10, help="number of epochs to training")

    parser.add_argument("--iters_per_epoch", type=int,
                        default=600, help="number of epochs to training")

    parser.add_argument("--batch_size", type=int, default=200,
                        help="batch size of the validation and testing data used during training")

    "Optimizer Parameters"
    parser.add_argument("--optimizer_type", dest="optimizer_type",
                        type=str, default="adam", help="adam or sgd")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument('--betas', nargs=2, metavar=('beta1', 'beta2'), default=(0.9, 0.999),
                        type=float, help="beta parameters for adam optimizer")

    parser.add_argument('--sgd_momentum', type=float,
                        default=0.0, help='momentum in sgd')

    parser.add_argument("--weight_decay", type=float,
                        default=.0, help="weight decay for discriminator")

    parser.add_argument('--scheduler_type', type=str, default="LinearPlateau",
                        help='learning rate scheduler: MultiStep, Linear, LinearPlateau')

    parser.add_argument('--lr_decay_factor', type=float, default=1e-4,
                        help='learning rate decay factor')

    parser.add_argument('--milestones', nargs="+", type=int, default=[120, 180],
                        help='milestones of learning rate decay')

    "Network Parameters"
    parser.add_argument('--model_structure', type=str,
                        default="mlp", help='model_structure',)

    parser.add_argument('--model_act_func', type=str,
                        default="lReLU[0.1]", help='model activation function',)

    parser.add_argument("--net_init_method", dest="net_init_method", type=str, default="kaiming",
                        help="network initialization method")

    parser.add_argument('--resamp_with_conv', action="store_true", default=False,
                        help='do down sampling with conv net in the network')

    parser.add_argument('--use_convshortcut', action="store_true", default=False,
                        help='use conv shortcut in the network')

    parser.add_argument('--dp_prob', type=float, default=0,
                        help='dropout probability')

    parser.add_argument("--network_normalization", dest="network_normalization", type=str, default=None,
                        help="network normalization")

    parser.add_argument(
        "--use_spectral_norm",
        default=False,
        action="store_true",
        help="whether use spectral norm",
    )

    "Data Preprocessing"
    parser.add_argument(
        "--uniform_dequantization",
        default=False,
        action="store_true",
        help="whether use uniform dequantization",
    )

    parser.add_argument('--gaussian_noise_std', type=float, default=None,
                        help='standard deviation of guassion noise in data augmentation')

    parser.add_argument('--random_horizontal_flip', action="store_true",
                        default=False,  help='random horizontal flip image')

    parser.add_argument("--data_augmentation", dest="data_augmentation", action="store_true",
                        help="do data augmentation")
    parser.add_argument("--no-data_augmentation",
                        dest="data_augmentation", action="store_false")
    parser.set_defaults(data_augmentation=True)

    parser.add_argument('--data_transform', type=str,
                        default="center_and_scale", help='transformation on the data',)

    "Sampling Parameters"
    parser.add_argument("--mala_reject",  dest="mala_reject", action="store_true",
                        default="no-reject", help="MALA sampler, rejection or no rejection")
    parser.add_argument("--no-mala_reject", dest="mala_reject", action="store_false")
    parser.set_defaults(mala_reject=False)

    parser.add_argument("--random_image_type", type=str, default="normal",
                        help="random images type, uniform, norma")

    parser.add_argument("--sample_steps", type=int,
                        default=40, help="sampling steps")

    parser.add_argument('--replay_buffer_size', type=check_none_or_int, default=None,
                        help='replay buffer size, if not speicified, we run short-run')

    parser.add_argument('--reinit_probs', type=float, default=0.0,
                        help='replay buffer, reinitialize probability')

    parser.add_argument('--start_reject_epochs', type=int,
                        default=10, help='start_reject_epochs')

    parser.add_argument('--longrun_test_freq', type=int,
                        default=50, help='long run log')

    parser.add_argument('--save_all_buffers', action="store_true", default=False,
                        help='save all buffer')

    parser.add_argument('--dynamic_sampling', action="store_true", default=False,
                        help='dynamically adjust sampling steps after warm-up iterations')
    parser.add_argument(
        "--adjust_step_size_freq", type=int, default=100, help="frequency of dynamically adjust sampling steps"
    )

    parser.add_argument(
        "--init_step_size_values", type=float, default=1e-2, help="intial step size in Langevin"
    )

    "Diffusion Parameters"
    parser.add_argument(
        "--num_diffusion_timesteps", type=int, default=6, help="number of time steps"
    )

    parser.add_argument(
        "--diffusion_schedule",
        type=str,
        default="linear",
        help="type of diffusion schedule: linear, sigmoid, quad, sqrtcumlinear",
    )

    parser.add_argument(
        "--diffusion_betas",
        nargs=2,
        metavar=("beta1", "beta2"),
        default=(1e-5, 5e-3),
        type=float,
        help="starting and ending betas of diffusion scheduler",
    )

    return parser


def add_ddpm_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp_dir", type=str, default="experiments/",
                        help="directory of the experiment")

    parser.add_argument("--data_dir", type=str,
                        default="data/", help="path to dataset")

    parser.add_argument("--t_seed", type=int, default=1234, help="manual seed")

    parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")

    parser.add_argument("--cuda", type=int, default="0", help="device-name")

    parser.add_argument('--save_all_annealed_models', action="store_true", default=False,
                        help='save all models when anneled learning rate')

    "Log Parameters"
    parser.add_argument("--log_dir", type=str, default="tb_log/",
                        help="path and prefix of tensorboard log file")

    parser.add_argument("--print_freq", type=int, default=100,
                        help="number of iterations for printing log during training")

    parser.add_argument('--write_tb_freq', type=int,
                        default=25, help='write tensorboard frequency')

    "Learning Parameters"
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs to training")

    parser.add_argument("--n_warm_epochs", type=int,
                        default=10, help="number of epochs to training")

    parser.add_argument("--iters_per_epoch", type=int,
                        default=600, help="number of epochs to training")

    parser.add_argument("--batch_size", type=int, default=200,
                        help="batch size of the validation and testing data used during training")

    "Optimizer Parameters"
    parser.add_argument("--optimizer_type", dest="optimizer_type",
                        type=str, default="adam", help="adam or sgd")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument('--betas', nargs=2, metavar=('beta1', 'beta2'), default=(0.9, 0.999),
                        type=float, help="beta parameters for adam optimizer")

    parser.add_argument('--sgd_momentum', type=float,
                        default=0.0, help='momentum in sgd')

    parser.add_argument("--weight_decay", type=float,
                        default=.0, help="weight decay for discriminator")

    parser.add_argument('--scheduler_type', type=str, default="LinearPlateau",
                        help='learning rate scheduler: MultiStep, Linear, LinearPlateau')

    parser.add_argument('--lr_decay_factor', type=float, default=1e-4,
                        help='learning rate decay factor')

    parser.add_argument('--milestones', nargs="+", type=int, default=[120, 180],
                        help='milestones of learning rate decay')

    "Network Parameters"
    parser.add_argument('--model_structure', type=str,
                        default="mlp", help='model_structure',)

    parser.add_argument('--model_act_func', type=str,
                        default="lReLU[0.1]", help='model activation function',)

    parser.add_argument("--net_init_method", dest="net_init_method", type=str, default="kaiming",
                        help="network initialization method")

    parser.add_argument('--resamp_with_conv', action="store_true", default=False,
                        help='do down sampling with conv net in the network')

    parser.add_argument('--use_convshortcut', action="store_true", default=False,
                        help='use conv shortcut in the network')

    parser.add_argument('--dp_prob', type=float, default=0,
                        help='dropout probability')

    parser.add_argument("--network_normalization", dest="network_normalization", type=str, default=None,
                        help="network normalization")

    parser.add_argument(
        "--use_spectral_norm",
        default=False,
        action="store_true",
        help="whether use spectral norm",
    )

    "Data Preprocessing"
    parser.add_argument(
        "--uniform_dequantization",
        default=False,
        action="store_true",
        help="whether use uniform dequantization",
    )

    parser.add_argument('--gaussian_noise_std', type=float, default=None,
                        help='standard deviation of guassion noise in data augmentation')

    parser.add_argument('--random_horizontal_flip', action="store_true",
                        default=False,  help='random horizontal flip image')

    parser.add_argument("--data_augmentation", dest="data_augmentation", action="store_true",
                        help="do data augmentation")
    parser.add_argument("--no-data_augmentation",
                        dest="data_augmentation", action="store_false")
    parser.set_defaults(data_augmentation=True)

    parser.add_argument('--data_transform', type=str,
                        default="center_and_scale", help='transformation on the data',)

    "Diffusion Parameters"
    parser.add_argument(
        "--num_diffusion_timesteps", type=int, default=1000, help="number of time steps"
    )

    parser.add_argument(
        "--diffusion_betas",
        nargs=2,
        metavar=("beta1", "beta2"),
        default=(1e-5, 5e-3),
        type=float,
        help="starting and ending betas of diffusion scheduler",
    )

    return parser


def add_drl_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp_dir", type=str, default="experiments/",
                        help="directory of the experiment")

    parser.add_argument("--data_dir", type=str,
                        default="data/", help="path to dataset")

    parser.add_argument("--t_seed", type=int, default=1234, help="manual seed")

    parser.add_argument("--saved_models_dir", type=str,
                        default="saved_models/net", help="prefix of saved models")

    parser.add_argument("--cuda", type=int, default="0", help="device-name")

    parser.add_argument('--save_all_annealed_models', action="store_true", default=False,
                        help='save all models when anneled learning rate')

    "Log Parameters"
    parser.add_argument("--log_dir", type=str, default="tb_log/",
                        help="path and prefix of tensorboard log file")

    parser.add_argument("--print_freq", type=int, default=100,
                        help="number of iterations for printing log during training")

    parser.add_argument('--write_tb_freq', type=int,
                        default=25, help='write tensorboard frequency')

    "Learning Parameters"
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs to training")

    parser.add_argument("--n_warm_epochs", type=int,
                        default=10, help="number of epochs to training")

    parser.add_argument("--iters_per_epoch", type=int,
                        default=600, help="number of epochs to training")

    parser.add_argument("--batch_size", type=int, default=200,
                        help="batch size of the validation and testing data used during training")

    "Optimizer Parameters"
    parser.add_argument("--optimizer_type", dest="optimizer_type",
                        type=str, default="adam", help="adam or sgd")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    parser.add_argument('--betas', nargs=2, metavar=('beta1', 'beta2'), default=(0.9, 0.999),
                        type=float, help="beta parameters for adam optimizer")

    parser.add_argument('--sgd_momentum', type=float,
                        default=0.0, help='momentum in sgd')

    parser.add_argument("--weight_decay", type=float,
                        default=.0, help="weight decay for discriminator")

    parser.add_argument('--scheduler_type', type=str, default="LinearPlateau",
                        help='learning rate scheduler: MultiStep, Linear, LinearPlateau')

    parser.add_argument('--lr_decay_factor', type=float, default=1e-4,
                        help='learning rate decay factor')

    parser.add_argument('--milestones', nargs="+", type=int, default=[120, 180],
                        help='milestones of learning rate decay')

    "Network Parameters"
    parser.add_argument('--model_structure', type=str,
                        default="mlp", help='model_structure',)

    parser.add_argument('--model_act_func', type=str,
                        default="lReLU[0.1]", help='model activation function',)

    parser.add_argument("--net_init_method", dest="net_init_method", type=str, default="kaiming",
                        help="network initialization method")

    parser.add_argument('--resamp_with_conv', action="store_true", default=False,
                        help='do down sampling with conv net in the network')

    parser.add_argument('--use_convshortcut', action="store_true", default=False,
                        help='use conv shortcut in the network')

    parser.add_argument('--dp_prob', type=float, default=0,
                        help='dropout probability')

    parser.add_argument("--network_normalization", dest="network_normalization", type=str, default=None,
                        help="network normalization")

    parser.add_argument(
        "--use_spectral_norm",
        default=False,
        action="store_true",
        help="whether use spectral norm",
    )
    "Sampling Parameters"
    parser.add_argument(
        "--b_factor",
        type=float,
        default=2e-4,
        help="step-size factor in Langevin sampling",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help="sampling steps"
        )

    "Data Preprocessing"
    parser.add_argument(
        "--uniform_dequantization",
        default=False,
        action="store_true",
        help="whether use uniform dequantization",
    )

    parser.add_argument('--gaussian_noise_std', type=float, default=None,
                        help='standard deviation of guassion noise in data augmentation')

    parser.add_argument('--random_horizontal_flip', action="store_true",
                        default=False,  help='random horizontal flip image')

    parser.add_argument("--data_augmentation", dest="data_augmentation", action="store_true",
                        help="do data augmentation")
    parser.add_argument("--no-data_augmentation",
                        dest="data_augmentation", action="store_false")
    parser.set_defaults(data_augmentation=True)

    parser.add_argument('--data_transform', type=str,
                        default="center_and_scale", help='transformation on the data',)

    "Diffusion Parameters"
    parser.add_argument(
        "--num_diffusion_timesteps", type=int, default=6, help="number of time steps"
    )

    parser.add_argument(
        "--diffusion_schedule",
        type=str,
        default="linear",
        help="type of diffusion schedule: linear, sigmoid, quad, sqrtcumlinear",
    )

    parser.add_argument(
        "--diffusion_betas",
        nargs=2,
        metavar=("beta1", "beta2"),
        default=(1e-4, 2e-2),
        type=float,
        help="starting and ending betas of diffusion scheduler",
    )

    return parser
