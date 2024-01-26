# P2IKT
Prior and Prediction Inverse Kernel Transformer for Single Image Defocus Deblurring (Accepted by AAAI-2024)
## Python Libriries 
torch==1.8.0+cu111  
torchvision==0.9.0+cu11  
albumentations==0.5.2  
opencv2==4.7.0  
tqdm==4.65.0  
sklearn==0.24.2  
pandas==2.0.0  
keras==2.12.0  
numpy==1.23.5  
lpips==0.1.4

## Test
The pretrained weight is already in the directory of pretrain_weight (P2IKT_SWA.pkl), you can directly use it to reproduce the result of our paper.  
Just modifying the data_dir for the testing in your environment.
## Train
Our model has two-stage training. 
### First stage training
We train the model using 384 x 384 images for 1500 epochs. So, we train the model with the following  commandline : python main.py --model_name "P2IKT" --mode "train" --data_dir 'you data path' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 384  --batch_size 4  
### Second stage training
Then our model are trained with 512 x 512 image s for 100th. So, we train the model with this commandline : python main.py --model_name "P2IKT" --mode "train" --data_dir 'you data path' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 512  --batch_size 4  --resume './weight/1500.pkl'(something like that, the weight of 1500th in last training stage)
### Training from the pretrained 1500th weights
If the researcher think the first stage training is too long, we also provide the pretrained 1500th weight of our model (model_1500.pth) in the directory of pretrained_weight (https://drive.google.com/file/d/1eqQFKlqDp1S4FDB1-_A49-CarGC6-kMt/view?usp=drive_link Please download the weight from this link), you can directly start from the second stage training by changing the path of --resume  
python main.py --model_name "P2IKT" --mode "train" --data_dir 'you data path' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 512  --batch_size 4  --resume './pretrained_weight/1500.pkl'



### Notice
Although we fixed the random_seed in our experiments, but nn.upsample(mode='bilinear') is nondeterministic (please refer to this link  https://github.com/pytorch/pytorch/issues/107999), So, the results of each training will be subtly different.


## Our code is free for the use of scientific research and non-commercial research
