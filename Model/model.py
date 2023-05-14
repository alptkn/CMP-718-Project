import torch
import torch.nn as nn



class ConvMainBlock(nn.Module):
    def __init__(self, inp, out, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, out, kernel_size, stride, padding ), nn.ReLU(True))

    def forward(self, input):
        x = self.conv(input)

        return x

class TransposeConv(nn.Module):
    def __init__(self, inp, out, kernel_size, stride, padding, out_padding):
        super().__init__()
        self.trans = nn.Sequential(nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, stride = stride, padding=padding, out_padding=out_padding))

    def forward(self, input):
        x = self.trans(input)

        return x

class MixUp(nn.Module):
    def __init__(self, m=-0.88):
        super().__init_()
        w = nn.parameter.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        mix_factor = self.sig(self.w)
        out = x1 * mix_factor.expaand_as(x1) + x2 *(1 -  mix_factor.expand_as(x2))

        return out
    

class DehazeNetwork(nn.Module):
    def __init__(self, inp, out, mc = 64):
        super().__init__()
        self.down1 = ConvMainBlock(inp, mc, kernel_size=7, stride=1, padding=0)
        self.down2 = ConvMainBlock(mc, mc*2, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvMainBlock(mc*2, mc*4, kernel_size=3, stride=2, padding=1)

        self.mix1 = MixUp(m=-1)
        self.mix2 = MixUp(m=-0.6)

        self.up1 = TransposeConv(mc*4, mc*2, kernel_size=3, stride=2, padding=1, out_padding=1)
        self.up2 = TransposeConv(mc*2, mc, kernel_size=3, stride=2, padding=1, out_padding=0)
        self.up3 = TransposeConv(mc, out, kernel_size=7, stride=1, padding=0, out_padding=0)
    
    def forward(self, input):
        x_down_1 = self.down1(input)
        x_down_2 = self.down2(x_down_1)
        x_down_3 = self.down3(x_down_2)

        """
        This is not complete. 
        Here we will add 6 FA blocks and 2 DFE block. 
        x_dcn_2 will be the output of the second DFE block. 
        """

        x_dcn_2 = "output of second DFE module"

        x_mix = self.mix1(x_down_3, x_dcn_2)
        x_up_1 = self.up1(x_mix)
        x_up_1_mix = self.mix2(x_down_2, x_up_1)
        x_up_2 = self.up2(x_up_1_mix)
        out = self.up3(x_up_2)

        return out

    

