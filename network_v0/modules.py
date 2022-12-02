import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import image_grid

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

       
class DilationConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationConv3x3, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class InterestPointModule(nn.Module):
    def __init__(self, is_test=False):
        super(InterestPointModule, self).__init__()
        self.is_test = is_test
        
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.maxpool2x2 = nn.MaxPool2d(2, 2)
        
        # score head
        self.score_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.score_norm = nn.BatchNorm2d(256)
        self.score_out = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        
        # location head
        self.loc_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.loc_norm = nn.BatchNorm2d(256)
        self.loc_out = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

        # descriptor out
        self.des_conv2 = DilationConv3x3(64, 256)
        self.des_conv3 = DilationConv3x3(128, 256)

        # cross_head:
        self.shift_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
                    
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, _, H, W = x.shape

        x = self.conv1(x)
        x = self.maxpool2x2(x)
        x2 = self.conv2(x)
        x = self.maxpool2x2(x2)
        x3 = self.conv3(x)
        x = self.maxpool2x2(x3)
        x = self.conv4(x)

        B, _, Hc, Wc = x.shape
        
        # score head
        score_x = self.score_out(self.relu(self.score_norm(self.score_conv(x))))
        aware = self.softmax(score_x[:, 0:2, :, :])
        score = score_x[:, 2, :, :].unsqueeze(1).sigmoid()
        
        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)
        
        # location head        
        coord_x = self.relu(self.loc_norm(self.loc_conv(x)))
        coord_cell = self.loc_out(coord_x).tanh()
        
        shift_ratio = self.shift_out(coord_x).sigmoid() * 2.0

        step = ((H/Hc)-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=coord_cell.dtype,
                                 device=coord_cell.device,
                                 ones=False, normalized=False).mul(H/Hc) + step

        coord_un = center_base.add(coord_cell.mul(shift_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)

        # descriptor block
        desc_block = []
        desc_block.append(self.des_conv2(x2))
        desc_block.append(self.des_conv3(x3))
        desc_block.append(aware)

        if self.is_test:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W-1)/2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            desc2 = torch.nn.functional.grid_sample(desc_block[0], coord_norm)         
            desc3 = torch.nn.functional.grid_sample(desc_block[1], coord_norm)
            aware = desc_block[2]
            
            desc = torch.mul(desc2, aware[:, 0, :, :]) + torch.mul(desc3, aware[:, 1, :, :])         
            desc = desc.div(torch.unsqueeze(torch.norm(desc, p=2, dim=1), 1))  # Divide by norm to normalize.

            return score, coord, desc

        return score, coord, desc_block


class CorrespondenceModule(nn.Module):
    def __init__(self, match_type='dual_softmax'):
        super(CorrespondenceModule, self).__init__()
        self.match_type = match_type

        if self.match_type == 'dual_softmax':
            self.temperature = 0.1
        else:
            raise NotImplementedError()
 
    def forward(self, source_desc, target_desc):
        b, c, h, w = source_desc.size()       
     
        source_desc = source_desc.div(torch.unsqueeze(torch.norm(source_desc, p=2, dim=1), 1)).view(b, -1, h*w)
        target_desc = target_desc.div(torch.unsqueeze(torch.norm(target_desc, p=2, dim=1), 1)).view(b, -1, h*w)

        if self.match_type == 'dual_softmax':
            sim_mat = torch.einsum("bcm, bcn -> bmn", source_desc, target_desc) / self.temperature
            confidence_matrix = F.softmax(sim_mat, 1) * F.softmax(sim_mat, 2)
        else:
            raise NotImplementedError()
        
        return confidence_matrix