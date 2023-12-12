import os
import torch
import argparse
from torch.backends import cudnn
import numpy as np 

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})
    #return {'Total': total_num, 'Trainable': trainable_num}
 
def main(args):
    # CUDNN
    cudnn.benchmark = False
    import random
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] =str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True     
    
   # if not os.path.exists('results/'):
   #     os.makedirs(args.model_save_dir, exist_ok=True)
   # if not os.path.exists('results/' + args.model_name + '/'):
   #     os.makedirs('results/' + args.model_name + '/', exist_ok=True)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
   # if not os.path.exists(args.result_dir):
   #     os.makedirs(args.result_dir, exist_ok=True)

    model = build_net(args.model_name,args.data_mode,args.is_training, args.image_size)
    get_parameter_number(model)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)

    elif args.mode == 'search':
        _search(model, args)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNet', type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='test', choices=['train', 'test', 'search'], type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--swa_epoch', default=500, type=int)
    parser.add_argument('--use_swa', default=True, type=bool)
    parser.add_argument('--data_mode', default='SP', type=str)   
    parser.add_argument('--dataset_mode', default='full_img_train', type=str)   
    parser.add_argument('--seed', default=10, type=int)   
    parser.add_argument('--image_size', default=256, type=int)   
                
    
    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    #parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)]) 
    parser.add_argument('--lr_steps', type=list, default=[500,1000,1500,2000,2500]) 
    parser.add_argument('--loss_mode', type=str, default='non_mask_loss')
    parser.add_argument('--is_training', type=bool, default=True)
    
    # Test
    parser.add_argument('--test_model', type=str, default='weights/MIMO-UNet.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('./training_weight/', args.model_name, args.data_mode, args.loss_mode, args.dataset_mode, 'weights/')
    #args.result_dir = os.path.join('multi_task_learning_results/', args.model_name, args.data_mode, args.dataset_mode,'result_image/')
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from models.network import build_net
    from train import _train
    from eval import _eval, _search

    main(args)
