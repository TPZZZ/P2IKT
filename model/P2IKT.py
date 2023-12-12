import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)
        base_channel = 64
        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, base_channel, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1))


        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(base_channel, 160, kernel_size=3, stride=2, padding=1),
            self.activation,
        )


    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx
        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx
        hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2


class Embeddings_output(nn.Module):
    def __init__(self):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(160, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )
        head_num = 4
        dim = 64

        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(dim+dim, dim, kernel_size=1, padding=0),
            self.activation,
        )



        self.de_block_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        
        self.de_block_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        
        self.de_block_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            
        self.de_block_4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        
        
        self.de_layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(dim+dim, dim, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.de_layer1_4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        
     #   self.de_layer1_1 = nn.Sequential(
    #        nn.Conv2d(32, 3, kernel_size=3, padding=1),
    #        self.activation
    #    )
    
    def forward(self, x, residual_1, residual_2):


        hx = self.de_layer3_1(x)

        hx = self.de_layer2_2(torch.cat((hx, residual_2), dim = 1))
        hx = self.activation(self.de_block_1(hx) + hx)
        hx = self.activation(self.de_block_2(hx) + hx)
        hx = self.activation(self.de_block_3(hx) + hx)
        hx = self.activation(self.de_block_4(hx) + hx)
      #  hx = self.de_block_5(hx)
       # hx = self.de_block_6(hx)
        hx = self.de_layer2_1(hx)

        hx = self.activation(self.de_layer1_1(torch.cat((hx, residual_1), dim = 1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.activation(self.de_layer1_3(hx) + hx)
        hx = self.activation(self.de_layer1_4(hx) + hx)
        
       # hx = self.de_layer1_1(hx)

        return hx

class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x


class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)
    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
        
        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x

class Inter_SA(nn.Module):
    def __init__(self,dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]
        
        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        
        horizontal_groups = horizontal_groups.view(3*B, H, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        
        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]


        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x

class Stripformer_1(nn.Module):
    def __init__(self):
        super(Stripformer_1, self).__init__()

        self.encoder = Embeddings()
        head_num = 4
        dim = 160
        self.Trans_block_1 = Intra_SA(dim, head_num)
        self.Trans_block_2 = Inter_SA(dim, head_num)
        self.Trans_block_3 = Intra_SA(dim, head_num)
        self.Trans_block_4 = Inter_SA(dim, head_num)
        self.Trans_block_5 = Intra_SA(dim, head_num)
        self.Trans_block_6 = Inter_SA(dim, head_num)
       # self.Trans_block_7 = Intra_SA(dim, head_num)
       # self.Trans_block_8 = Inter_SA(dim, head_num)
       # self.Trans_block_9 = Intra_SA(dim, head_num)
       # self.Trans_block_10 = Inter_SA(dim, head_num)
       # self.Trans_block_11 = Intra_SA(dim, head_num)
       # self.Trans_block_12 = Inter_SA(dim, head_num)
        self.decoder = Embeddings_output()


    def forward(self, x):

        hx, residual_1, residual_2 = self.encoder(x)
        hx = self.Trans_block_1(hx)
        hx = self.Trans_block_2(hx)
        hx = self.Trans_block_3(hx)
        hx = self.Trans_block_4(hx)
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)
      #  hx = self.Trans_block_7(hx)
      #  hx = self.Trans_block_8(hx)
      #  hx = self.Trans_block_9(hx)
      #  hx = self.Trans_block_10(hx)
      #  hx = self.Trans_block_11(hx)
      #  hx = self.Trans_block_12(hx)
        hx = self.decoder(hx, residual_1, residual_2)

        return hx 

class SumLayer(nn.Module):
    def __init__(self, num_kernels=21,iterations=1, trainable=False, factor=2):
        super(SumLayer, self).__init__()
        self.conv = nn.Conv2d(factor * (num_kernels + 1) * 3 * iterations, 3, 1)

    def forward(self, x):
        return self.conv(x)

class MultiplyLayer1(nn.Module):
    def __init__(self, iterations=1):
        super(MultiplyLayer1, self).__init__()
        self.iterations = iterations
    def forward(self, x, y):
        if self.iterations == 1:
            
            return x * torch.cat([y, y, y], dim=1)
        elif self.iterations == 2:
            return x * torch.cat([y, y, y, y, y, y], dim=1)            
        elif self.iterations == 3:
            return x * torch.cat([y, y, y, y, y, y, y, y, y], dim=1)            
        

class MultiplyLayer(nn.Module):
    def __init__(self, iterations=1):
        super(MultiplyLayer, self).__init__()
        self.ml = MultiplyLayer1(iterations=iterations)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        return torch.cat([self.ml(x[:, :c // 2], y[:, :c1 // 2]), self.ml(x[:, c // 2:], y[:, c1 // 2:])], dim=1)

class BasicConv1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False, dilation=1, if_padding=True, separable=False):
        super(BasicConv1, self).__init__()
        if bias and norm:
            bias = False
        if if_padding == True:
            padding = (kernel_size-1) * dilation // 2
        else:
            padding = kernel_size // 2
        layers = list()
        if transpose:
            if if_padding == True:
                padding = kernel_size * dilation // 2 -1
            else:
                padding = kernel_size // 2-1
                
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation))
        elif separable:
            layers.append(
                nn.Conv2d(in_channel, out_channel, 1, padding=0, stride=stride, bias=bias, dilation=dilation)
                )
            layers.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation, groups=out_channel))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation),
                )

        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Predictor(nn.Module):
    def __init__(self, in_nc=1, nf=128, size=27 * 27, use_bias=False):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=11, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU6(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(nf, size, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU6(),
            nn.AdaptiveAvgPool2d((3, 1)),
            nn.Conv2d(size, size, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Softmax(),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        #self.seBlock = SEBlock(size)

    def forward(self, input, a, b):
        conv = self.ConvNet(input)
       # print(conv.size())
        #flat = self.globalPooling(conv)
        #print(flat.size())
        #re = self.seBlock(flat)
        return conv.view(-1, 3, a, b)  # torch size: [B, code_len]
        

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()
    
def gen_inverse_gaussian_kernel_1(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().cpu().numpy()
    inverse_kernel = np.fft.fft2(_2D_window)
    #inverse_kernel = 1 / (inverse_kernel)
    inverse_kernel = np.conj(inverse_kernel) / (np.abs(inverse_kernel) ** 2 + 0.01) #wierner
    inverse_kernel = np.fft.ifft2(inverse_kernel)
    inverse_kernel = np.abs(inverse_kernel)
    inverse_kernel_torch = torch.from_numpy(inverse_kernel).unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(inverse_kernel_torch.expand(1, 1, window_size, window_size).contiguous())
    return window
        

class GaussianAdaBlurLayer10_inverse_1_out(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianAdaBlurLayer10_inverse_1_out, self).__init__()
        self.channels = channels
        kernel_size = 3
        weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
        for i in range(num_kernels):
            pad = int((max_kernel_size - kernel_size) / 2)
            weight[i + 1] = (F.pad(gen_inverse_gaussian_kernel_1(kernel_size, sigma=0.25 * (i + 1)).cuda(),
                                   [pad, pad, pad, pad])).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)
        #print(weight.size())
        kernel = np.repeat(weight, self.channels, axis=0).cuda()
        if mode == 'TG':
            self.weight = kernel
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels * 3, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)
        #print(self.weight.size())
        #self.adptive_weight = nn.Parameter(data=torch.randn((num_kernels+1) * 3, 1, max_kernel_size, max_kernel_size),
        #                               requires_grad=True)
        self.adaption_conv = Predictor(3,128,size=num_kernels * num_kernels)
        self.predicted_kernel = num_kernels
        
        self.transform_conv = BasicConv1(3, (num_kernels+1)*3, kernel_size=3, stride=1, relu=False, norm=False, dilation=1, separable=False)
        
        self.fuse_conv = BasicConv1((num_kernels+1)*3*2, (num_kernels+1)*3, kernel_size=3, stride=1, relu=False, norm=False, dilation=1, separable=False)
    
        self.Convsout = BasicConv1(3, 3, kernel_size=3, stride=1, relu=False, norm=False, dilation=1, separable=False)
    
        
    def __call__(self, x, dilation=1):

        x_gauss = F.conv2d(x, self.weight, padding=(self.predicted_kernel-1)*dilation//2, groups=self.channels, dilation=dilation)

        x_adaptive_kernel = self.adaption_conv(x,self.predicted_kernel,self.predicted_kernel)
        #print(x_adaptive_kernel.size())
        output = torch.zeros_like(x)
        for i in range(x_adaptive_kernel.size()[0]):
          A_1 = x[i:i+1,:,:,:]
          B_1 = x_adaptive_kernel[i:i+1,:,:,:]
         # B_1 = B_1.view(3,-1,21,21)
          output_1 = F.conv2d(A_1, B_1, padding=(self.predicted_kernel-1)*dilation//2, groups=1, dilation=dilation)          
          output[i:i+1,:,:,:] = output_1
        #print(x_adaptive_feature.size())
        ax_output = self.Convsout(x) 
        x_adaptive_feature = self.transform_conv(output)
        
      #  x_adaption_attn = self.adaption_attn(x)
        
        x = self.fuse_conv(torch.cat([x_gauss, x_adaptive_feature], dim=1))
        return x, ax_output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())
        
class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out, kernelsize=3):
        super(conv_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernelsize, stride=1, padding=int((kernelsize - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)       
        
class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block1(ch_in, ch_out)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, hidden_state)
        y = self.upsample(h)
        return self.sigmoid((y * x_res) + y) * 2 - 1, h, c

class P2IKT_model(nn.Module):
    def __init__(self, kernel_mode='FG', num_gaussian_kernels=5, gaussian_kernel_size=5):
        super(type(self), self).__init__()
        super().__init__()
        self.GCM = GaussianAdaBlurLayer10_inverse_1_out(num_gaussian_kernels, gaussian_kernel_size, kernel_mode)
        '''backbone'''
        self.basemodel = Stripformer_1()
        '''APU'''
        self.APU = SqueezeAttentionBlock(64, 2 * (num_gaussian_kernels + 1))
        '''entry-wise multiplication'''
        self.MultiplyLayer = MultiplyLayer()
        '''summation'''
        self.SumLayer = SumLayer(num_gaussian_kernels)
        self.SumLayer1 =  nn.Conv2d(6,3,1)
        
    def forward_step(self, input_blurry, last_output, hidden_state, dilation):
        ''' Gaussian Reblurring '''
        gy,ax_out_y = self.GCM(input_blurry, dilation)
        gx,ax_out_x = self.GCM(last_output, dilation)

        '''Feature Extraction'''
        f4_1 = self.basemodel(input_blurry) 
        '''Weight Maps Generation'''
        weights, h, c = self.APU(f4_1, hidden_state)

        '''Entry-wise Multiplication and Summation'''
        result = self.SumLayer(self.MultiplyLayer(torch.cat([gy, gx], dim=1), weights))
        ax_out = self.SumLayer1(torch.cat([ax_out_x,ax_out_y],dim=1))
        return result, ax_out, h, c

    def forward(self, input_blur_256, input_blur_128=None, input_blur_64=None):
        input_blur_128 = F.interpolate(input_blur_256, scale_factor=0.5)
        input_blur_64 = F.interpolate(input_blur_128, scale_factor=0.5)
        h, c = self.APU.conv_atten.init_hidden(
            input_blur_64.shape[0],
            (input_blur_64.shape[-2] // 2, input_blur_64.shape[-1] // 2))
        """The forward process"""
        '''scale 1'''
        db64, ax_out_64, h, c = self.forward_step(input_blur_64, input_blur_64, (h, c), 1)
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        '''scale 2'''
        db128,ax_out_128, h, c = self.forward_step(input_blur_128, F.upsample(db64, scale_factor=2, mode='bilinear'), (h, c), 1)
        h = F.upsample(h, scale_factor=2, mode='bilinear')
        c = F.upsample(c, scale_factor=2, mode='bilinear')
        '''scale 3'''
        db256,ax_out_256, _, _ = self.forward_step(input_blur_256, F.upsample(db128, scale_factor=2, mode='bilinear'), (h, c), 1)
        return  db64, db128, ax_out_256,db256 