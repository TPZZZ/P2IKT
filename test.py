import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
from torchvision.transforms import functional as F
from data import valid_dataloader, test_dataloader
from utils import Adder
from natsort import natsorted
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda() 
from ptflops import get_model_complexity_info

from models.network import build_net
import numpy as np
from skimage.metrics import structural_similarity
def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True)
log10 = np.log(10)
import time
from sklearn.metrics import mean_absolute_error
def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])
    
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    return total_num

import cv2   


    
def test(dataset, model_name_list, experimental_name=None, save_image=False, downscale=False, data_mode='SP_MSL',dataset_mode='full_img_train'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dataset = 'realdof'
    if dataset == 'dpdd':
        data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/dd_dp_dataset_canon/dd_dp_dataset_png/test_c/'
    elif dataset == 'realdof':
        if downscale :
          data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/realdof_downscale_2/'
          dataset = 'realdof_downscale_2'
        else:        
          data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/realdof/'
    elif dataset == 'CUHK':
        data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/CUHK/'
    elif dataset == 'PixelDP':
        if downscale:
            data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/pixelDP_resize/'
            dataset = 'PixelDP_resize'
        else:
            data_dir = '/u/home/tangp/tangp/ImageRestoration/ImageDeblurring/defocus/pixelDP/'        

    dataloader = test_dataloader(data_dir, batch_size=1, num_workers=0, data_mode=data_mode,dataset_mode=dataset_mode)
    #save_image = False

    RE_PSNR_list = []
    RE_SSIM_list = []
    RE_LPIPS_list = []
    RE_MAE_list  = []
    RE_TIME_list = []
    Model_Parm_list = []
    

    
    for model_name in model_name_list:
        print('{} Start to Test'.format(model_name))
        if save_image:
            result_dir = './ICCV_without_value/{}/{}/{}/'.format(experimental_name,model_name,dataset)
            os.makedirs(result_dir,exist_ok=True)

   
        model = build_net(model_name, data_mode='SP')
        model_para = get_parameter_number(model)
        
 


        model_weight_dir = './pretrained_weight/{}_SWA.pkl'.format(model_name)
        state_dict = torch.load(model_weight_dir) 
        model.load_state_dict(state_dict['model']) 

        
  
        Model_Parm_list.append(model_para) 
        model.to(device)  
        model.eval()


        with torch.no_grad():
            for iter_idx, data in enumerate(dataloader):
 
                  input_img, label_img, name = data
                  print(input_img.size())

                  input_img = input_img
                  label_img = label_img
                 
                  input_img = input_img.to(device)


                  tm = time.time()

                  _ = model(input_img)                    


 
                  _ = time.time() - tm
 
                  if iter_idx == 20:
                      break

            psnr_adder = Adder()
            ssim_adder = Adder()
            lpips_adder = Adder()
            mae_adder  = Adder()
            time_adder = Adder() 
            for iter_idx, data in enumerate(dataloader):

                input_img, label_img, name = data


                input_img = input_img.to(device)

                
                tm = time.time()
                pred = model(input_img)[-1]
                elapsed = time.time() - tm
            
                time_adder(elapsed)

                pred_clip = torch.clamp(pred, 0, 1)
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()    

                psnr = peak_signal_noise_ratio(pred_numpy.transpose(1,2,0), label_numpy.transpose(1,2,0), data_range=1)   
                lpip_score = loss_fn_alex(pred_clip.cuda(), label_img.cuda(),normalize=True).item()   
                ssim_score = ssim(pred_numpy.transpose(1,2,0),label_numpy.transpose(1,2,0)) 
                MAE_score = MAE(label_numpy.transpose(1,2,0), pred_numpy.transpose(1,2,0)) 
                psnr_adder(psnr)
                ssim_adder(ssim_score)
                lpips_adder(lpip_score)
                mae_adder(MAE_score)
                print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed)) 

                if save_image:
                   # print(name)[]
                    #save_name = os.path.join(result_dir, '{}_{:.3f}_{:.3f}.png'.format(name[0].split('.')[0], psnr, ssim_score))
                    save_name = os.path.join(result_dir, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred.save(save_name)

        print("PSNR: {:.4f}".format(psnr_adder.average()))
        print("SSIM: {:.4f}".format(ssim_adder.average()))
        print('LPIPS: {:.4f}'.format(lpips_adder.average())) 
        print('MAE: {:.4f}'.format(mae_adder.average())) 

        print("Average time: %f" % time_adder.average())

        RE_PSNR_list.append(psnr_adder.average())
        RE_SSIM_list.append(ssim_adder.average())
        RE_LPIPS_list.append(lpips_adder.average())
        RE_MAE_list.append(mae_adder.average())
        RE_TIME_list.append(time_adder.average())
    
    print(RE_PSNR_list)
    print(RE_SSIM_list)
    print(RE_LPIPS_list)
    print(RE_MAE_list)
    print(RE_TIME_list)
    print(Model_Parm_list) 




model_name_list = ['P2IKT']
experimental_name = 'albation_study'


test('dpdd',model_name_list,experimental_name, 0, 0,'SP_MSL')  
test('realdof',model_name_list,experimental_name, 0, 1,'SP_MSL')
