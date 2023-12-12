import os
import torch
import IFAN.LPIPS as LPIPS
from data import train_dataloader
from utils import Adder, Timer, check_lr
import torchvision
from valid import _valid
import torch.nn.functional as F
from torchcontrib.optim import SWA
from pytorch_msssim import ssim
import torch.nn as nn
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer

LPIP_loss = LPIPS.PerceptualLoss(model='net-lin',net='alex', gpu_ids = [torch.cuda.current_device()]).to(torch.device('cuda'))
for param in LPIP_loss.parameters():
    param.requires_grad_(False)



def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion_fft = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    CE = nn.CrossEntropyLoss()
              
    optimizer = torch.optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, data_mode=args.data_mode, dataset_mode=args.dataset_mode,image_size=args.image_size)
    max_iter = len(dataloader)
    opt = SWA(optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epoch, eta_min=1e-6)
    opt.step()
    scheduler.step()
    
    #sche

    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    #writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):
        

        if epoch_idx + 1 >= 1:
            lr = 1e-4

        if epoch_idx + 1 >= 1000:
            lr = 1e-4 / 2 
        if epoch_idx + 1 >= 1500:
            lr = 1e-4 / 2 / 2 
        if epoch_idx + 1 >= 2000:
            lr = 1e-4 / 2 / 2 / 2 / 2
        if epoch_idx + 1 >= 2500:
            lr = 1e-4 / 2 / 2 / 2 / 2 / 2    
            
        if args.image_size == 384 :
            lr = lr * 2

        if lr < 1e-6:
            lr = 1e-6

        optimizer.param_groups[0]['lr'] = lr
        opt.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
        print(opt.param_groups[0]['lr'])
        print(optimizer.param_groups[0]['lr'])
        
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            if args.data_mode == "SP" or args.data_mode == 'SP_MSL':
              input_img, label_img = batch_data
              input_img = input_img.to(device)
              label_img = label_img.to(device)
              opt.zero_grad()
              pred_img = model(input_img)  
                  
              if args.loss_mode == 'fft_loss_out':
                label_fft3 = torch.fft.fft(label_img)
                pred_fft3 = torch.fft.fft(pred_img[-1])

                dist = LPIP_loss.forward(pred_img[-1],label_img)
                lpip_loss = criterion_fft(torch.zeros_like(dist).to(torch.device('cuda')), dist)                           
                
                l4 = criterion(pred_img[-1], label_img)      
                loss_content =  l4   
                ax_loss = criterion(pred_img[-2], label_img)  
                f3 = criterion_fft(pred_fft3, label_fft3)                
                loss_fft =  f3 

                loss = loss_content + 0.2 * loss_fft  + 0.2 * lpip_loss   + 0.05*ax_loss

                loss = loss_content + loss_fft  + 0.2 * lpip_loss + 0.2*(1-torch.mean(ssim(pred_img[-1], label_img, data_range=1, size_average=False)))

              
            loss.backward()
            opt.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                
            if epoch_idx > (args.num_epoch  - args.swa_epoch) and epoch_idx % 1 == 0:                
                opt.update_swa()

                
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')

        torch.save({'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))

            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
    opt.swap_swa_sgd()
    swa_save_name = os.path.join(args.model_save_dir, 'SWA.pkl')
    torch.save({'model': model.state_dict()}, swa_save_name)
    
