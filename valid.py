import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import IFAN.LPIPS as LPIPS
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
#from models.MIMOUNet import build_net



def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0, data_mode=args.data_mode, dataset_mode=args.dataset_mode)
    model.eval()
    psnr_adder = Adder()
    l_psnr_adder = Adder()
    r_psnr_adder = Adder()
    f_psnr_adder = Adder()
    p_re_psnr_adder = Adder()
    lpips_score_adder = Adder()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            if args.data_mode == "SP" or args.data_mode == 'SP_MSL'  or args.data_mode == 'Patch_SP_MSL' :
            
                input_img, label_img, name = data

                input_img = input_img.to(device)
                label_img = label_img.to(device)


                pred = model(input_img)
                pred_clip = torch.clamp(pred[-1], 0, 1)
                p_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()

                psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
                

            psnr_adder(psnr)     
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
