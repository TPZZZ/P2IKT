import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
#LPIPSN = LPIPS.PerceptualLoss(model='net-lin',net='alex').to(torch.device('cuda'))

def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True)
log10 = np.log(10)

def patch_prediction(full_image):
    image_patch_list = torch.zeros(35,3,256,256).cuda()
    A = full_image[0,:,:,:]
    #print(A.size())
    size = 256
    idx = 0
    for i in range(0,5):
      for j in range(0,7):
          if (i+1)*size > A.size()[1]:
              image_patch_list[idx,:,:,:] = A[:,A.size()[1]-size:A.size()[1],j*size:(j+1)*size]
          if (i+1)*size > A.size()[1]:
              image_patch_list[idx,:,:,:] = A[:,A.size()[1]-size:A.size()[1],j*size:(j+1)*size]
          else:
              image_patch_list[idx,:,:,:] = A[:,i*size:(i+1)*size,j*size:(j+1)*size]         
          idx+=1
    return image_patch_list

def merge_patch(patch_image):
  A = torch.zeros(1,3,720,1280)
  B = patch_image
  #print(A.size())
  #print(B.size())
  size = 256
  idx = 0
  for i in range(0,5):
      for j in range(0,7):
          
        #  print(A[0,:,A.size()[2]-size:A.size()[2],j*size:(j+1)*size].size())
          if (i+1)*size > A.size()[2]:
              A[0,:,A.size()[2]-size:A.size()[2],j*size:(j+1)*size] = B[idx,:,:,:]
          else:
              A[0,:,i*size:(i+1)*size,j*size:(j+1)*size] = B[idx,:,:,:]
          idx+=1
  return A




def compute_psnr(x, label, max_diff):
    assert max_diff in [255, 1, 2]
    if max_diff == 255:
        x = x.clamp(0, 255)
    elif max_diff == 1:
        x = x.clamp(0, 1)
    elif max_diff == 2:
        x = x.clamp(-1, 1)

    mse = ((x - label) ** 2).mean()
    return 10 * torch.log(max_diff ** 2 / mse) / log10


def _eval(model, args):
    state_dict = torch.load(args.test_model) 
    model.load_state_dict(state_dict['model'])
   # model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0, data_mode=args.data_mode)
    torch.cuda.empty_cache()
    adder = Adder()

    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        l_psnr_adder = Adder()
        r_psnr_adder = Adder()
        f_psnr_adder = Adder()
        p_re_psnr_adder = Adder()
        ssim_adder = Adder()
        lpips_adder = Adder()
        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            if args.data_mode == 'SP' or args.data_mode == 'realdof':
              input_img, label_img = data
              
              input_img = input_img[:,:,:400,:608]
              label_img = label_img[:,:,:400,:608]
              #print(input_img.size())
              input_img = input_img.to(device)
              tm = time.time()
              _ = model(input_img)
              _ = time.time() - tm
  
              if iter_idx == 20:
                  break
            else:
              input_img_l, input_img_r, label_img = data
              input_img_l = input_img_l.to(device)
              input_img_r = input_img_r.to(device)
              
              tm = time.time()
              _ = model((input_img_l,input_img_r))
              _ = time.time() - tm
  
              if iter_idx == 20:
                  break            
        # Main Evaluation
        l_img_dir = os.path.join(args.result_dir, 'left/')
        r_img_dir = os.path.join(args.result_dir, 'right/')
        c_img_dir = os.path.join(args.result_dir, 'combine/')
        
        print(l_img_dir)
        os.makedirs(l_img_dir,exist_ok=True)
        os.makedirs(r_img_dir,exist_ok=True)
        os.makedirs(c_img_dir,exist_ok=True)
        
        print(len(dataloader))
        for iter_idx, data in enumerate(dataloader):
        
            if args.data_mode == 'SP' or args.data_mode == 'realdof':
              if args.model_name != "GRDC2MIMOUNet" and args.model_name != "CSDC2MIMOUNet_6" and args.model_name != "CSDC2MIMOUNet_7" and args.model_name != "CSDC2MIMOUNet_8" and args.model_name != 'CSDC2MIMOUNet_9':
                input_img, label_img = data
                #print(input_img.size())
                input_img = input_img[:,:,:400,:608]
                label_img = label_img[:,:,:400,:608]
                  
                input_img = input_img.to(device)
    
                tm = time.time()
                pred = model(input_img)[-1]
                elapsed = time.time() - tm
                adder(elapsed)
    
                pred_clip = torch.clamp(pred, 0, 1)
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()
              else:
                input_img, label_img = data
                input_img = input_img[:,:,:400,:608]
                label_img = label_img[:,:,:400,:608]
                    
                input_img = input_img.to(device)
    
                tm = time.time()
                pred = model(input_img)
                elapsed = time.time() - tm
                adder(elapsed)
    
                pred_clip = torch.clamp(pred[0], 0, 1)
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()
                
                pred_re_clip = torch.clamp(pred[-1], 0, 1)
                p_re_numpy = pred_re_clip.squeeze(0).cpu().numpy()
                
                p_re_psnr = peak_signal_noise_ratio(p_re_numpy, label_numpy, data_range=1)    
                f_psnr = peak_signal_noise_ratio(0.9*p_re_numpy+0.1*pred_numpy, label_numpy, data_range=1)    
                
                f_psnr_adder(f_psnr)
                p_re_psnr_adder(p_re_psnr)  
                                                         

            else:
              if args.model_name != 'MMDC2MIMOUNet' and  args.model_name != 'MMDC2MIMOUNet1' :
              
                input_img_l, input_img_r, label_img = data
    
                input_img_l = input_img_l.to(device)
                input_img_r = input_img_r.to(device)

    
                tm = time.time()
                pred = model((input_img_l,input_img_r))[2]
                elapsed = time.time() - tm
                adder(elapsed)
                pred_clip = torch.clamp(pred, 0, 1)
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()                
              else:
                input_img_l, input_img_r, label_img = data
    
                input_img_l = input_img_l.to(device)
                input_img_r = input_img_r.to(device)

                tm = time.time()
                pred = model((input_img_l, input_img_r))   
                elapsed = time.time() - tm
                adder(elapsed)
                
                left_pred_clip = torch.clamp(pred[2][0], 0, 1)
                right_pred_clip = torch.clamp(pred[2][1], 0, 1)
                pred_clip = torch.clamp(pred[2][2], 0, 1)
                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                l_p_numpy = left_pred_clip.squeeze(0).cpu().numpy()
                r_p_numpy = right_pred_clip.squeeze(0).cpu().numpy()
                
                label_numpy = label_img.squeeze(0).cpu().numpy()
                l_psnr = peak_signal_noise_ratio(l_p_numpy, label_numpy, data_range=1)   
                r_psnr = peak_signal_noise_ratio(r_p_numpy, label_numpy, data_range=1)   
                f_psnr = peak_signal_noise_ratio(0.25*r_p_numpy+0.5*l_p_numpy+0.25*pred_numpy, label_numpy, data_range=1)   
                
                l_psnr_adder(l_psnr)    
                r_psnr_adder(r_psnr)   
                f_psnr_adder(f_psnr)   
                
                np.save(os.path.join(l_img_dir, '{}.npy'.format(iter_idx)),l_p_numpy)                  
                np.save(os.path.join(r_img_dir, '{}.npy'.format(iter_idx)),r_p_numpy)                  
                np.save(os.path.join(c_img_dir, '{}.npy'.format(iter_idx)),pred_numpy)                  
            
            #print(pred_numpy.transpose(1,2,0).shape)
           # print(label_numpy.shape)
            if args.model_name != 'CSDC2MIMOUNet_6' and args.model_name != 'CSDC2MIMOUNet_7':
              ssim_score = ssim(pred_numpy.transpose(1,2,0),label_numpy.transpose(1,2,0))
            else:
              ssim_score = ssim((0.9*p_re_numpy+0.1*pred_numpy).transpose(1,2,0),label_numpy.transpose(1,2,0))            
            #lpips_score =loss_fn_alex(input_img, label_img.to(device))).cpu().numpy()[0][0][0][0]
       
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
      
            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)


            psnr_adder(psnr)
            ssim_adder(ssim_score)
            #lpips_adder(lpips_score)
            
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))    
            #print('%d iter PSNR_de: %.2f time: %f' % (iter_idx + 1, psnr_de, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.3f dB' % (ssim_adder.average()))
       # print('The average LPIPS is %.4f dB' % (lpips_adder.average()))
        
      #  print('The average PSNR_de is %.2f dB' % (psnr_adder_de.average()))
        if args.model_name == "MMDC2MIMOUNet" or args.model_name == "MMDC2MIMOUNet1":
          print("C-PSNR: {}".format(psnr_adder.average()))
          print("L-PSNR: {}".format(l_psnr_adder.average()))
          print("R-PSNR: {}".format(r_psnr_adder.average()))
          print("F-PSNR: {}".format(f_psnr_adder.average()))

        if args.model_name == 'GRDC2MIMOUNet' or args.model_name == "CSDC2MIMOUNet_6" or args.model_name == 'CSDC2MIMOUNet_7' or args.model_name == 'CSDC2MIMOUNet_8' or args.model_name == 'CSDC2MIMOUNet_9':
          print("P-PSNR: {}".format(psnr_adder.average()))
          print("P-RE-PSNR: {}".format(p_re_psnr_adder.average()))
          #print("R-PSNR: {}".format(r_psnr_adder.average()))
          print("F-PSNR: {}".format(f_psnr_adder.average()))      
              
        print("Average time: %f" % adder.average())

       

def _search(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0, data_mode=args.data_mode)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            if args.data_mode == 'SP':
              input_img, label_img = data
              input_img = input_img.to(device)
              tm = time.time()
              _ = model(input_img)
              _ = time.time() - tm
  
              if iter_idx == 20:
                  break
            else:
              input_img_l, input_img_r, label_img = data
              input_img_l = input_img_l.to(device)
              input_img_r = input_img_r.to(device)
              
              tm = time.time()
              _ = model((input_img_l,input_img_r))
              _ = time.time() - tm
  
              if iter_idx == 20:
                  break  
                            
        # Main Evaluation
        best_PSNR = 0
        best_weight = None
        num = 0
        
        l_img_list = []
        r_img_list = []
        c_img_list = []
        label_list = []
        
        for iter_idx, data in enumerate(dataloader):
            input_img_l, input_img_r, label_img = data

            input_img_l = input_img_l.to(device)
            input_img_r = input_img_r.to(device)

            pred = model((input_img_l, input_img_r))   
            
            left_pred_clip = torch.clamp(pred[2][0], 0, 1)
            right_pred_clip = torch.clamp(pred[2][1], 0, 1)
            pred_clip = torch.clamp(pred[2][2], 0, 1)
            
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            l_p_numpy = left_pred_clip.squeeze(0).cpu().numpy()
            r_p_numpy = right_pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy() 
            
            l_img_list.append(l_p_numpy)
            r_img_list.append(r_p_numpy)
            c_img_list.append(pred_numpy)
            label_list.append(label_numpy)
            
            print('%d iter' % (iter_idx + 1))     
                  
        l_img_list = np.array(l_img_list)
        r_img_list = np.array(r_img_list)
        c_img_list = np.array(c_img_list)
        label_list = np.array(label_list)
        
        
        for i in range(1,10,1):
          for j in range(1,10,1):
            for z in range(1,10,1):
              num+=1
              total = i + j + z
            #  if total > 10:
            #    continue
              i = i / total
              j = j / total
              z = z / total
              f_psnr_adder = Adder() 
              psnr_adder = []
              for img_idx in range(l_img_list.shape[0]):
              
              #f_psnr_adder = Adder()        
                f_psnr = peak_signal_noise_ratio(z*r_img_list[img_idx]+j*l_img_list[img_idx]+i*c_img_list[img_idx], label_list[img_idx], data_range=1)   
                psnr_adder.append(f_psnr)
                f_psnr_adder(f_psnr)   
              print(num)
         #     print('The average PSNR is %.2f dB' % (f_psnr_adder.average()))
         #     print('The Weight is i: {}, j: {}, z: {}'.format(i,j,z))  
              if np.mean(psnr_adder) > best_PSNR:
                  best_PSNR = np.mean(psnr_adder)
                  best_weight = [i,j,z]
                  print('==========================================================')
                  print('The average PSNR is %.2f dB' % (np.mean(psnr_adder)))
                  print('The Weight is i: {}, j: {}, z: {}'.format(i,j,z))
                  
