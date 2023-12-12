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
## Test
The pretrained weight is already in the directory of pretrain_weight, you can directly use it to reproduce the result of our paper.  
Just modifying the data_dir for the testing in your environment.
## Train
Our model has two-stage training. Firstly, we train the model using 384 x 384 images for 1500 epochs. So, we train the model with the following  commandline : python main.py --model_name "P2IKT" --mode "train" --data_dir '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/dd_dp_dataset_canon/dd_dp_dataset_png/' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 384  --batch_size 4  
Then our model are trained with 512 x 512 image s for 100th. So, we train the model with this commandline : python main.py --model_name "P2IKT" --mode "train" --data_dir '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/dd_dp_dataset_canon/dd_dp_dataset_png/' --gpu "3" --swa_epoch 100 --num_epoch 1600 --data_mode "SP_MSL" --seed 10 --loss_mode 'fft_loss_out' --image_size 512  --batch_size 4  --resume './weight/1500.pth'(something like that, the weight of 1500th in last training stage)

