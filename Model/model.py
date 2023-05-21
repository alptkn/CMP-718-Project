import torch
import torch.nn as nn
from torchvision import models


class ConvMainBlock(nn.Module):
    def __init__(self, inp, out, kernel_size, stride, padding):
        super(ConvMainBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, out, kernel_size, stride, padding ), nn.ReLU(True))

    def forward(self, input):
        x = self.conv(input)

        return x

class TransposeConv(nn.Module):
    def __init__(self, inp, out, kernel_size, stride, padding, out_padding):
        super(TransposeConv, self).__init__()
        self.trans = nn.Sequential(nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, stride = stride, padding=padding, out_padding=out_padding))

    def forward(self, input):
        x = self.trans(input)

        return x

class MixUp(nn.Module):
    def __init__(self, m=-0.88):
        super(MixUp, self).__init_()
        w = nn.parameter.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        mix_factor = self.sig(self.w)
        out = x1 * mix_factor.expaand_as(x1) + x2 *(1 -  mix_factor.expand_as(x2))

        return out
    

class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        x = self.avg_pooling(input)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sig(x)

        return torch.mul(input, x)

class PixelAttention(nn.Module):
    def __init__(self, channel):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sig(x)

        return torch.mul(input, x)

class FABlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(FABlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(dim)
        self.pa = PixelAttention(dim)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        y = input + x
        y = self.conv2(y)
        y = self.ca(y)
        y = self.pa(y)

        return torch.add(input, y)


class DehazeNetwork(nn.Module):
    def __init__(self, inp, out, mc = 64):
        super(DehazeNetwork, self).__init__()
        self.down1 = ConvMainBlock(inp, mc, kernel_size=7, stride=1, padding=0)
        self.down2 = ConvMainBlock(mc, mc*2, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvMainBlock(mc*2, mc*4, kernel_size=3, stride=2, padding=1)

        self.mix1 = MixUp(m=-1)
        self.mix2 = MixUp(m=-0.6)
        self.fa_block = FABlock(mc*4, kernel_size=3)

        self.up1 = TransposeConv(mc*4, mc*2, kernel_size=3, stride=2, padding=1, out_padding=1)
        self.up2 = TransposeConv(mc*2, mc, kernel_size=3, stride=2, padding=1, out_padding=0)
        self.up3 = TransposeConv(mc, out, kernel_size=7, stride=1, padding=0, out_padding=0)
    
    def forward(self, input):
        x_down_1 = self.down1(input)
        x_down_2 = self.down2(x_down_1)
        x_down_3 = self.down3(x_down_2)

        x1 = self.fa_block(x_down_3)
        x2 = self.fa_block(x1)
        x3 = self.fa_block(x2)
        x4 = self.fa_block(x3)
        x5 = self.fa_block(x4)
        x6 = self.fa_block(x5)


       

        x_dcn_2 = "output of second DFE module"

        x_mix = self.mix1(x_down_3, x_dcn_2)
        x_up_1 = self.up1(x_mix)
        x_up_1_mix = self.mix2(x_down_2, x_up_1)
        x_up_2 = self.up2(x_up_1_mix)
        out = self.up3(x_up_2)

        return out

    
    """ if __name__ == "__main__":
        t = torch.Tensor([2.05, 2.03, 3.8, 2.29])
        t2 = torch.Tensor([7,7,7,7])
        print(t + t2)
        print(torch.add(t, t2)) 
         """

