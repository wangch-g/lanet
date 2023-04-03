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
    def __init__(self, is_test=True):
        super(PointModel, self).__init__()
        self.is_test = is_test
        self.interestpoint_module = InterestPointModule(is_test=self.is_test)
        self.correspondence_module = CorrespondenceModule()
        self.norm_rgb = tvf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
  
    def forward(self, *args):
        if self.is_test:
            img = args[0]
            img = self.norm_rgb(img)
            score, coord, desc = self.interestpoint_module(img)
            return score, coord, desc
        else:
            source_score, source_coord, source_desc_block = self.interestpoint_module(args[0])
            target_score, target_coord, target_desc_block = self.interestpoint_module(args[1])

            B, _, H, W = args[0].shape
            B, _, hc, wc = source_score.shape
            device = source_score.device

            # Normalize the coordinates from ([0, h], [0, w]) to ([0, 1], [0, 1]).
            source_coord_norm = source_coord.clone()
            source_coord_norm[:, 0] = (source_coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
            source_coord_norm[:, 1] = (source_coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
            source_coord_norm = source_coord_norm.permute(0, 2, 3, 1)

            target_coord_norm = target_coord.clone()
            target_coord_norm[:, 0] = (target_coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
            target_coord_norm[:, 1] = (target_coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
            target_coord_norm = target_coord_norm.permute(0, 2, 3, 1)
            
            target_coord_warped_norm = warp_homography_batch(source_coord_norm, args[2])
            target_coord_warped = target_coord_warped_norm.clone()
        
            # de-normlize the coordinates
            target_coord_warped[:, :, :, 0] = (target_coord_warped[:, :, :, 0] + 1) * (float(W - 1) / 2.)
            target_coord_warped[:, :, :, 1] = (target_coord_warped[:, :, :, 1] + 1) * (float(H - 1) / 2.)
            target_coord_warped = target_coord_warped.permute(0, 3, 1, 2)

            # Border mask
            border_mask_ori = torch.ones(B, hc, wc)
            border_mask_ori[:, 0] = 0
            border_mask_ori[:, hc - 1] = 0
            border_mask_ori[:, :, 0] = 0
            border_mask_ori[:, :, wc - 1] = 0
            border_mask_ori = border_mask_ori.gt(1e-3).to(device)

            oob_mask2 = target_coord_warped_norm[:, :, :, 0].lt(1) & target_coord_warped_norm[:, :, :, 0].gt(-1) & target_coord_warped_norm[:, :, :, 1].lt(1) & target_coord_warped_norm[:, :, :, 1].gt(-1)
            border_mask = border_mask_ori & oob_mask2

            # score
            target_score_warped = torch.nn.functional.grid_sample(target_score, target_coord_warped_norm.detach(), align_corners=False)

            # descriptor
            source_desc2 = torch.nn.functional.grid_sample(source_desc_block[0], source_coord_norm.detach())
            source_desc3 = torch.nn.functional.grid_sample(source_desc_block[1], source_coord_norm.detach())
            source_aware = source_desc_block[2]
            source_desc = torch.mul(source_desc2, source_aware[:, 0, :, :].unsqueeze(1).contiguous()) + torch.mul(source_desc3, source_aware[:, 1, :, :].unsqueeze(1).contiguous())

            target_desc2 = torch.nn.functional.grid_sample(target_desc_block[0], target_coord_norm.detach())
            target_desc3 = torch.nn.functional.grid_sample(target_desc_block[1], target_coord_norm.detach())
            target_aware = target_desc_block[2]
            target_desc = torch.mul(target_desc2, target_aware[:, 0, :, :].unsqueeze(1).contiguous()) + torch.mul(target_desc3, target_aware[:, 1, :, :].unsqueeze(1).contiguous())
            
            target_desc2_warped = torch.nn.functional.grid_sample(target_desc_block[0], target_coord_warped_norm.detach())
            target_desc3_warped = torch.nn.functional.grid_sample(target_desc_block[1], target_coord_warped_norm.detach())
            target_aware_warped = torch.nn.functional.grid_sample(target_desc_block[2], target_coord_warped_norm.detach())
            target_desc_warped = torch.mul(target_desc2_warped, target_aware_warped[:, 0, :, :].unsqueeze(1).contiguous()) + torch.mul(target_desc3_warped, target_aware_warped[:, 1, :, :].unsqueeze(1).contiguous())
            
            confidence_matrix = self.correspondence_module(source_desc, target_desc)
            confidence_matrix = torch.clamp(confidence_matrix, 1e-12, 1 - 1e-12)
            
            output = {
                'source_score': source_score,
                'source_coord': source_coord,
                'source_desc': source_desc,
                'source_aware': source_aware,
                'target_score': target_score,
                'target_coord': target_coord,
                'target_score_warped': target_score_warped,
                'target_coord_warped': target_coord_warped,
                'target_desc_warped': target_desc_warped,
                'target_aware_warped': target_aware_warped,
                'border_mask': border_mask,
                'confidence_matrix': confidence_matrix
            }
        
            return output
