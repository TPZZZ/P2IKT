#!/bin/bash
#
#SBATCH --job-name=P2IKT
#SBATCH --output=./P2IKT.txt
#SBATCH --error=./P2IKT.err
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=300:00:00    
#SBATCH --mem-per-cpu=4000  
       
#srun source activate pytorch
#srun python test_demo.py 

srun python main.py --model_name "P2IKT" --mode "train" --data_dir '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/dd_dp_dataset_canon/dd_dp_dataset_png/' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 384  --batch_size 4 



