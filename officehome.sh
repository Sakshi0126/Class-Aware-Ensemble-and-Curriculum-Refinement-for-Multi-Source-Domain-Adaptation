#!/bin/bash
#SBATCH --job-name=officehome-loss-new
#SBATCH --partition=cse-gpu-all
#SBATCH --nodelist=dgx-a100-02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00

module load anaconda3/2024.02-1

# Activate the Conda environment
source activate csr-env


srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:1 python3 main-ot-sh.py --name 'officehome' --src 'Art' 'Clipart' 'Real_World' --tar 'Product' --batch-size 32 --warmup-steps 0 --student-wait-steps 0 --wait_step 100 --bottleneck 'linear' --total-steps 10001 --num-classes 65 --tar_weight 1.5 --tar_threshold 0.9 --tar_alignment_weight 0.5 --src_tar_weight 0.0 --consis_tar --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15380 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/officehome-product-new-loss.log 2>&1 | tee ./officehome_loss_new_product.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:1 python3 main-ot-sh.py --name 'officehome' --src 'Product' 'Clipart' 'Real_World' --tar 'Art' --batch-size 32 --warmup-steps 0 --student-wait-steps 0 --wait_step 100 --bottleneck 'linear' --total-steps 10001 --num-classes 65 --tar_weight 1.5 --tar_threshold 0.9 --tar_alignment_weight 0.5 --src_tar_weight 0.0 --consis_tar --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15381 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/officehome-art-new-loss.log 2>&1 | tee ./officehome_loss_new_art.log

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:1 python3 main-ot-sh.py --name 'officehome' --src 'Art' 'Clipart' 'Product' --tar 'Real_World' --batch-size 32 --warmup-steps 0 --student-wait-steps 0 --wait_step 100 --bottleneck 'linear' --total-steps 10001 --num-classes 65 --tar_weight 1.5 --tar_threshold 0.9 --tar_alignment_weight 0.5 --src_tar_weight 0.0 --consis_tar --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15382 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/officehome-realworld-new-loss.log 2>&1 | tee ./officehome_loss_new_realworld.log 

srun --partition=cse-gpu-all --nodelist=dgx-a100-02 --gres=gpu:1 python3 main-ot-sh.py --name 'officehome' --src 'Art' 'Product' 'Real_World' --tar 'Clipart' --batch-size 32 --warmup-steps 0 --student-wait-steps 0 --wait_step 100 --bottleneck 'linear' --total-steps 10001 --num-classes 65 --tar_weight 1.5 --tar_threshold 0.9 --tar_alignment_weight 0.5 --src_tar_weight 0.0 --consis_tar --consis_weight 0.0 --ext_weight 0.0 --res_weight 0.01 --align_mode 1 --ext_mode 0 --seed 8 --amp --use_weight_pred --weight_tau 1.0 --port 15383 --grad-clip 10.0 --ot_alpha 1.0 --ot_beta 0.5 --ot_gamma 0.5 --word2vec_path 'GoogleNews-vectors-negative300.bin' --gamma_explore 0.5 --gamma_mid 0.5 --gamma_refine 0.5 --log_file_name ./logs/officehome-clipart-new-loss.log 2>&1 | tee ./officehome_loss_new_clipart.log
