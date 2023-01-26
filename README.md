# Persistently Trained, Diffusion-assisted Energy-based Models (DAEBM)


This repo contains the implementation for the paper **Persistently Trained, Diffusion-assisted Energy-based Models.**


## Requirements
Run the following to install the necessary python packages for the code. Code is tested with python=3.10.
```
pip install -r requirements.txt
```

## Usage
### Image experiments on MNIST and FashionMNIST
Set `--main_dir` to be `./mnist` or `./fashionmnist` to conduct experiments on **MNIST** or **FashionMNIST**. Datasets will be automatically downloaded. The following commands are examples to run differeent methods on **MNIST**.
#### DAEBM
##### Training
```angular2
python src/train-DAEBM.py --main_dir ./mnist --exp_dir experiments/cnn_DAEBM/exp1 --lr 1e-5  \
                  --betas 0.0 0.999 --optimizer_type adam --batch_size 200 --n_epochs 200 \
                  --scheduler_type Linear --milestones 120 --lr_decay_factor 1e-5 \
                  --data_transform center_and_scale --no-data_augmentation \
                  --model_structure  ResNetTemb[128,1,1,2,2,2] --model_act_func SiLU --use_convshortcut \
                  --no-mala_reject --sample_steps 40 --random_image_type normal --replay_buffer_size 50000 \
                  --dynamic_sampling --init_step_size_values 0.01 --start_reject_epochs 10 \
                  --num_diffusion_timesteps 50 --diffusion_schedule sqrtcumlinear --diffusion_betas 2e-4 2e-2 \
                  --save_all_annealed_models --cuda 0
```
##### Post-sampling
Start 100 chains in parallel and run 10000 MGMS traisitions. The number of Langevin steps within a MGMS transition is the same as in training.
```
python src/post-sampling.py --main_dir ./mnist --pexp_dir experiments/cnn_DAEBM/exp1 --sampling_chains 100 \
        --total_iters 100000 --stop_a_chain_M 50 --print_freq 500 --cuda 0
```

#### EBM
Set `--reinit_probs` to be `0`, `0.05`, `1` to train EBM with persistent initialization, hybrid initialization, and noise initialization, respectively.
```angular2
python src/train-EBM.py --main_dir ./fashionmnist --exp_dir experiments/cnn_EBM/exp1 --lr 2e-5  \
                  --betas 0.0 0.999 --optimizer_type adam --batch_size 200 --n_epochs 200 \
                  --scheduler_type Linear --milestones 120 --lr_decay_factor 1e-5 \
                  --data_transform gaussian_center_and_scale --gaussian_noise_std 0.03 --no-data_augmentation \
                  --model_structure  ResNetTemb[128,1,1,2,2,2] --model_act_func SiLU --use_convshortcut \
                  --no-mala_reject --sample_steps 40 --random_image_type normal --replay_buffer_size 50000 \
                  --mala_eps 0.005 --mala_std None --reinit_probs 0 \
                  --save_all_annealed_models --cuda 0
```

#### DRL
This implementation is based on the official Tensorflow version [here](https://github.com/ruiqigao/recovery_likelihood).
```
python src/train-GAO-DRL.py --main_dir ./fashionmnist --exp_dir experiments/cnn_DRL/exp1 --lr 2e-4  \
                  --betas 0.0 0.999 --optimizer_type adam --batch_size 200 --n_epochs 200 \
                  --scheduler_type LinearPlateau --milestones 120 180 --lr_decay_factor 1e-5 \
                  --data_transform center_and_scale --no-data_augmentation \
                  --model_structure  ResNetTemb[128,1,1,2,2,2] --model_act_func SiLU --use_convshortcut \
                  --num_diffusion_timesteps 50 --diffusion_schedule linear --diffusion_betas 1e-4 2e-2 \
                  --sample_steps 40 --b_factor 2e-2 --use_spectral_norm --cuda 0
```


#### cDDPM
This implementation is based on the official PyTorch version   [here](https://github.com/yang-song/score_sde_pytorch).
```
python src/train-cDDPM.py --main_dir ./fashionmnist --exp_dir experiments/cnn_DDPM/exp1 --lr 2e-4  \
                  --betas 0.0 0.999 --optimizer_type adam --batch_size 200 --n_epochs 1000 \
                  --scheduler_type MultiStep --milestones 940 460 480 --lr_decay_factor 0.1 \
                  --data_transform center_and_scale --no-data_augmentation \
                  --model_structure UNet[128,1,1,2,2,2] --model_act_func SiLU \
                  --num_diffusion_timesteps 1000 --diffusion_betas 0.1 20 --cuda 0
```


###  Illustrative experiements
To replicate the four rings example, please refer to commands in [`script_toy.sh`](./script_toy.sh). The usage is similar to the above codes in image experiments. To replicate the Gaussian 1D example, use the R code [`Gaussian1D.R`](./Gaussian1D/Gaussian1D.R).


## Acknowledgement
The implementation has referred to GitHub repos [ebm-anatomy](https://github.com/point0bar1/ebm-anatomy), [recovery_likelihood](https://github.com/yang-song/score_sde_pytorch), and [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
