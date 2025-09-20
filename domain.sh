#!/bin/bash
#SBATCH --job-name=domainnet
#SBATCH --partition=cse-gpu-all
#SBATCH --nodelist=dgx-a100-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# Load the Anaconda module
module load anaconda3/2024.02-1

# Activate the Conda environment
source activate csr-env


srun --partition=cse-gpu-all --nodelist=dgx-v100-01 --gres=gpu:2 python3 main.py --name 'DomainNet' --src 'quickdraw' 'infograph' 'painting' 'real' 'clipart' --tar 'sketch'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 40001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0  --port 15688 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' 2>&1 | tee ./domainnet_ot_sketch.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:2 python3 main-ot-sh.py --name 'DomainNet' --src 'quickdraw' 'infograph' 'painting' 'real' 'sketch' --tar 'clipart'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 40001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0  --port 15610 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/domainnet-clipart-new-loss.log 2>&1 | tee ./domainnet-new-loss-clipart.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:2 python3 main-ot-sh.py --name 'DomainNet' --src 'quickdraw' 'infograph' 'painting' 'sketch' 'clipart' --tar 'real'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 50001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15611 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/domainnet-real-new-loss.log 2>&1 | tee ./domainnet-new-loss-real.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:2 python3 main-ot-sh.py --name 'DomainNet' --src 'quickdraw' 'infograph' 'real' 'clipart' 'sketch' --tar 'painting'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 50001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0  --port 15612 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/domainnet-painting-new-loss.log 2>&1 | tee ./domainnet-new-loss-painting.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:2 python3 main-ot-sh.py --name 'DomainNet' --src 'quickdraw' 'painting' 'real' 'clipart' 'sketch' --tar 'infograph'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 40001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15613 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/domainnet-infograph-new-loss.log 2>&1 | tee ./domainnet-new-loss-infograph.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:2 python3 main-ot-sh.py --name 'DomainNet' --src 'quickdraw' 'infograph' 'painting' 'real' 'clipart' --tar 'sketch'  --batch-size 32 --backbone 'resnet101' --warmup-steps 0 --student-wait-steps 0 --eval-step 750 --wait_step 1500 --bottleneck 'linear' --total-steps 40001 --num-classes 345 --tar_weight 1.0 --tar_threshold 0.6 --tar_alignment_weight 0.7 --src_tar_weight 0.0 --consis_tar --box_warm_iter 1000 --bt 0.1 --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0  --port 15614 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/domainnet-sketch-new-loss.log 2>&1 | tee ./domainnet-new-loss-sketch.log

