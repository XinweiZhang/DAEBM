## DAEBM
python toy/src/train-toy-DAEBM.py --exp_dir experiments/DAEBM/exp1 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.0 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 200 \
                  --scheduler_type MultiStep --milestones 140 160 180 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyTembModel --model_act_func Softplus \
                  --sample_steps 40 --replay_buffer_size 10000 --b_factor 1e-1 \
                  --diffusion_betas 1e-2 3e-1  --num_diffusion_timesteps 6 --diffusion_schedule sqrtcumlinear \
                  --start_reject_epochs 1 --dynamic_sampling --cuda 0 &> /dev/null &

## EBM persistent
python toy/src/train-toy-EBM.py --exp_dir experiments/EBM/exp1 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.0 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 200 \
                  --scheduler_type MultiStep --milestones 140 160 180 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyModel --model_act_func ReLU \
                  --no-mala_reject --sample_steps 40 --random_image_type normal --replay_buffer_size 10000 \
                  --mala_eps 0.005 --mala_std None --reinit_probs 0  --cuda 0 &> /dev/null &

## EBM hybrid
python toy/src/train-toy-EBM.py --exp_dir experiments/EBM/exp2 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.0 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 200 \
                  --scheduler_type MultiStep --milestones 140 160 180 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyModel --model_act_func ReLU \
                  --no-mala_reject --sample_steps 200 --random_image_type normal --replay_buffer_size 10000 \
                  --mala_eps 0.005 --mala_std None --reinit_probs 0.05 --refresh --cuda 0 &> /dev/null &


## EBM Noise
python toy/src/train-toy-EBM.py --exp_dir experiments/EBM/exp2 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.0 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 200 \
                  --scheduler_type MultiStep --milestones 140 160 180 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyModel --model_act_func ReLU \
                  --no-mala_reject --sample_steps 40000 --random_image_type normal --replay_buffer_size 10000 \
                  --mala_eps 0.005 --mala_std None --reinit_probs 1 --refresh --cuda 0 &> /dev/null &

## EBM CD
python toy/src/train-toy-EBM.py --exp_dir experiments/EBM/exp2 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.0 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 200 \
                  --scheduler_type MultiStep --milestones 140 160 180 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyModel --model_act_func ReLU \
                  --no-mala_reject --sample_steps 40 --random_image_type normal --replay_buffer_size 10000 \
                  --mala_eps 0.005 --mala_std None --use_cd --cuda 0 &> /dev/null &

## DRL
python toy/src/train-toy-DRL.py --exp_dir experiments/DRL/exp1 --toy_sd 0.01 --toy_radius 1 \
                  --betas 0.9 0.999 --optimizer_type adam --lr 5e-4 --n_epochs 500 \
                  --scheduler_type MultiStep --milestones 350 400 450 --lr_decay_factor 0.1 \
                  --batch_size 100 --n_warm_epochs 1 --model_structure ToyTembModel --model_act_func Softplus \
                  --diffusion_betas 1e-2 3e-1  --num_diffusion_timesteps 6 --diffusion_schedule sqrtcumlinear \
                  --sample_steps 40 --b_factor 2e-2 --cuda 0 &> /dev/null &
