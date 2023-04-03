import torch
import torch.nn as nn
import torchvision.transforms as tvf

from .modules import InterestPointModule, CorrespondenceModule

def warp_homography_batch(sources, homographies):
    """
    Batch warp keypoints given homographies. From https://github.com/TRI-ML/KP2D.

    Parameters
    ----------
    sources: torch.Tensor (B,H,W,C)
        Keypoints vector.
    homographies: torch.Tensor (B,3,3)
        Homographies.

    Returns
    -------
    warped_sources: torch.Tensor (B,H,W,C)
        Warped keypoints vector.
    """
    B, H, W, _ = sources.shape
    warped_sources = []
    for b in range(B):
        source = sources[b].clone()
        source = source.view(-1,2)
        '''
        [X,    [M11, M12, M13    [x,    M11*x + M12*y + M13           [M11, M12      [M13,
         Y,  =  M21, M22, M23  *  y, =  M21*x + M22*y + M23 = [x, y] * M21, M22    +  M23,
         Z]     M31, M32, M33]    1]    M31*x + M32*y + M33            M31, M32].T    M33]
        '''
        source = torch.addmm(homographies[b,:,2], source, homographies[b,:,:2].t())
        source.mul_(1/source[:,2].unsqueeze(1))
        source = source[:,:2].contiguous().view(H,W,2)
        warped_sources.append(source)
    return torch.stack(warped_sources, dim=0)

 
class PointModel(nn.Module):
    def __init__(self, is_test=False):
        super(PointModel, self).__init__()
        self.is_test = is_test
        self.interestpoint_module = InterestPointModule(is_test=self.is_test)
        self.correspondence_module = CorrespondenceModule()
        self.norm_rgb = tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    def forward(self, *args):
        img = args[0]
        img = self.norm_rgb(img)
        score, coord, desc = self.interestpoint_module(img)
        return score, coord, desc
