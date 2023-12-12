import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import random
import cv2
from skimage.metrics import peak_signal_noise_ratio

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True, data_mode='SP', dataset_mode='crop_train', image_size=256):
    if dataset_mode == "full_img_train" or dataset_mode == 'LF_DPDD':
      image_dir = os.path.join(path, 'train_c/')
    elif dataset_mode == "LF":
      image_dir = os.path.join(path, 'training/')
    
    
    if data_mode == 'SP_MSL':
      transform = None
      if use_transform:
          transform = PairCompose(
              [
                  PairRandomCrop(image_size),
                  PairRandomHorizontalFilp(),
                  PairToTensor()
              ]
          )      
          
      dataloader = DataLoader(
          DeblurDataset1(image_dir, transform=transform, image_size=image_size, dataset_mode=dataset_mode),
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers,
          pin_memory=True
      )  
         

    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0, data_mode='SP', dataset_mode='full_img_train'):

    if data_mode == 'SP_MSL':
      dataloader = DataLoader(
          DeblurDataset1(os.path.join(path), is_test=True, dataset_mode=dataset_mode),
          batch_size=batch_size,
          shuffle=False,
          num_workers=num_workers
      )
        
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0, data_mode='SP',dataset_mode=None):
          
    if data_mode == 'SP_MSL' or data_mode == 'Patch_SP_MSL' or data_mode == 'Patch_SP_MSL_Class':
      data_path = os.path.join(path, 'test_c/')
      if dataset_mode == 'LF':
        data_path = os.path.join(path, 'testing/')
      elif dataset_mode == "Gopro":
        data_path = os.path.join(path, 'test/')
                        
      dataloader = DataLoader(
          DeblurDataset1(data_path,is_test=True,dataset_mode=dataset_mode),
          batch_size=batch_size,
          shuffle=False,
          num_workers=num_workers)

        
    return dataloader

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)







class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = []


        for i in sorted(os.listdir(os.path.join(image_dir, 'source/'))):
            if i.split('.')[1] != "png" and  i.split('.')[1] != "jpg" and  i.split('.')[1] != "bmp" :
              continue
            self.image_list.append(i)
        self.label_list = []
        for i in sorted(os.listdir(os.path.join(image_dir, 'target/'))):
            if i.split('.')[1] != "png" and  i.split('.')[1] != "jpg" and  i.split('.')[1] != "bmp":
              continue
            self.label_list.append(i)
            
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'source', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.label_list[idx]))
  
        #if self.is_test:
        #  pass
        #else:
        #  image = image.resize((256,256))
        #  label = label.resize((256,256))
        
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'bmp']:
                raise ValueError



class DeblurDataset1(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False,image_size=256, dataset_mode='full_img_train'):
        self.image_dir = image_dir
        self.image_list = []
        self.image_size = image_size
        self.dataset_mode = dataset_mode
        if self.dataset_mode == 'full_img_train':
          self.blur_prefix = 'source/'
          self.sharp_prefix = 'target/'
        else:
          self.blur_prefix = 'source/'
          self.sharp_prefix = 'target/'
        
        for i in sorted(os.listdir(os.path.join(image_dir, self.blur_prefix))):
            if i.split('.')[1] != "png" and  i.split('.')[1] != "jpg" and  i.split('.')[1] != "bmp" :
              continue
            self.image_list.append(i)
        self.label_list = []
        for i in sorted(os.listdir(os.path.join(image_dir, self.sharp_prefix))):
            if i.split('.')[1] != "png" and  i.split('.')[1] != "jpg" and  i.split('.')[1] != "bmp":
              continue
            self.label_list.append(i)
            
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.dataset_mode == 'full_img_train':
            image = Image.open(os.path.join(self.image_dir, 'source', self.image_list[idx]))
            label = Image.open(os.path.join(self.image_dir, 'target', self.label_list[idx]))
        else:
            image = Image.open(os.path.join(self.image_dir, 'source', self.image_list[idx]))
            label = Image.open(os.path.join(self.image_dir, 'target', self.label_list[idx])) 
                   
        if self.transform:
            ps = self.image_size
            #image, label = self.transform(image, label)
            inp_img = image
            
            tar_img = label
            w,h = label.size
            #h,w = label.size()[:2]
            #print(label.size())
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
    
            # Reflect Pad in case image is smaller than patch_size
            if padw!=0 or padh!=0:
                inp_img = F.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
                tar_img = F.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
            

            inp_img = F.to_tensor(inp_img)
            tar_img = F.to_tensor(tar_img)

            hh, ww = tar_img.shape[1], tar_img.shape[2]
    
            rr     = random.randint(0, hh-ps)
            cc     = random.randint(0, ww-ps)
            aug    = random.randint(0, 8)
    
            # Crop patch
            inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]          
            tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
           # print(inp_img)
          #  psnr = peak_signal_noise_ratio(np.array(inp_img), np.array(tar_img), data_range=1)  
          #  print(psnr)
            
            if aug==1: 
                inp_img = inp_img.flip(1)
                tar_img = tar_img.flip(1)
            elif aug==2:
                inp_img = inp_img.flip(2)
                tar_img = tar_img.flip(2)
            elif aug==3:
                inp_img = torch.rot90(inp_img,dims=(1,2))   
                tar_img = torch.rot90(tar_img,dims=(1,2))
            elif aug==4:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=2)            
                tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            elif aug==5:
                inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
                tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            elif aug==6:
                inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
                tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            elif aug==7:
                inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))    
                tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
                
            output_image = inp_img
            output_label = tar_img
            
        else:
        
       
            output_image = F.to_tensor(image)
            output_label = F.to_tensor(label)
        

            
        if self.is_test:
            name = self.image_list[idx]
            
            return output_image, output_label, name
            
        return output_image, output_label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError



