import torch

def build_descriptor_loss(source_des, target_des, tar_points_un, top_kk=None, relax_field=4, eval_only=False):
    """
    Desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.

    Parameters
    ----------
    source_des: torch.Tensor (B,256,H/8,W/8)
        Source image descriptors.
    target_des: torch.Tensor (B,256,H/8,W/8)
        Target image descriptors.
    source_points: torch.Tensor (B,H/8,W/8,2) 
        Source image keypoints 
    tar_points: torch.Tensor (B,H/8,W/8,2)
        Target image keypoints 
    tar_points_un: torch.Tensor (B,2,H/8,W/8)
        Target image keypoints unnormalized 
    eval_only: bool
        Computes only recall without the loss.
    Returns
    -------
    loss: torch.Tensor
        Descriptor loss.
    recall: torch.Tensor
        Descriptor match recall.
    """
    device = source_des.device
    loss = 0
    batch_size = source_des.size(0)
    recall = 0.

    relax_field_size = [relax_field]
    margins          = [1.0]
    weights          = [1.0]

    isource_dense = top_kk is None

    for b_id in range(batch_size):

        if isource_dense:
            ref_desc = source_des[b_id].squeeze().view(256, -1)
            tar_desc = target_des[b_id].squeeze().view(256, -1)
            tar_points_raw = tar_points_un[b_id].view(2, -1)
        else:
            top_k = top_kk[b_id].squeeze()

            n_feat = top_k.sum().item()
            if n_feat < 20:
                continue

            ref_desc = source_des[b_id].squeeze()[:, top_k]
            tar_desc = target_des[b_id].squeeze()[:, top_k]         
            tar_points_raw = tar_points_un[b_id][:, top_k]

        # Compute dense descriptor distance matrix and find nearest neighbor
        ref_desc = ref_desc.div(torch.norm(ref_desc, p=2, dim=0))
        tar_desc = tar_desc.div(torch.norm(tar_desc, p=2, dim=0))
        dmat = torch.mm(ref_desc.t(), tar_desc)

        dmat = torch.sqrt(2 - 2 * torch.clamp(dmat, min=-1, max=1))
        _, idx = torch.sort(dmat, dim=1)


        # Compute triplet loss and recall
        for pyramid in range(len(relax_field_size)):

            candidates = idx.t()

            match_k_x = tar_points_raw[0, candidates]
            match_k_y = tar_points_raw[1, candidates]

            tru_x = tar_points_raw[0]
            tru_y = tar_points_raw[1]

            if pyramid == 0:
                correct2 = (abs(match_k_x[0]-tru_x) == 0) & (abs(match_k_y[0]-tru_y) == 0)
                correct2_cnt = correct2.float().sum()
                recall += float(1.0 / batch_size) * (float(correct2_cnt) / float( ref_desc.size(1)))

            if eval_only:
                continue
            correct_k = (abs(match_k_x - tru_x) <= relax_field_size[pyramid]) & (abs(match_k_y - tru_y) <= relax_field_size[pyramid])

            incorrect_index = torch.arange(start=correct_k.shape[0]-1, end=-1, step=-1).unsqueeze(1).repeat(1,correct_k.shape[1]).to(device)
            incorrect_first = torch.argmax(incorrect_index * (1 - correct_k.long()), dim=0)

            incorrect_first_index = candidates.gather(0, incorrect_first.unsqueeze(0)).squeeze()

            anchor_var = ref_desc
            posource_var = tar_desc
            neg_var = tar_desc[:, incorrect_first_index]

            loss += float(1.0 / batch_size) * torch.nn.functional.triplet_margin_loss(anchor_var.t(), posource_var.t(), neg_var.t(), margin=margins[pyramid]).mul(weights[pyramid])

    return loss, recall


class KeypointLoss(object):
    """
    Loss function class encapsulating the location loss, the descriptor loss, and the score loss.
    """
    def __init__(self, config):
        self.score_weight = config.score_weight
        self.loc_weight = config.loc_weight
        self.desc_weight = config.desc_weight
        self.corres_weight = config.corres_weight
        self.corres_threshold = config.corres_threshold
        
    def __call__(self, data):
        B, _, hc, wc = data['source_score'].shape
        
        loc_mat_abs = torch.abs(data['target_coord_warped'].view(B, 2, -1).unsqueeze(3) - data['target_coord'].view(B, 2, -1).unsqueeze(2))
        l2_dist_loc_mat = torch.norm(loc_mat_abs, p=2, dim=1)
        l2_dist_loc_min, l2_dist_loc_min_index = l2_dist_loc_mat.min(dim=2)

        # construct pseudo ground truth matching matrix
        loc_min_mat = torch.repeat_interleave(l2_dist_loc_min.unsqueeze(dim=-1), repeats=l2_dist_loc_mat.shape[-1], dim=-1)
        pos_mask = l2_dist_loc_mat.eq(loc_min_mat) & l2_dist_loc_mat.le(1.)
        neg_mask = l2_dist_loc_mat.ge(4.)

        pos_corres = - torch.log(data['confidence_matrix'][pos_mask])
        neg_corres = - torch.log(1.0 - data['confidence_matrix'][neg_mask])
        corres_loss = pos_corres.mean() + 5e5 * neg_corres.mean()

        # corresponding distance threshold is 4
        dist_norm_valid_mask = l2_dist_loc_min.lt(self.corres_threshold) & data['border_mask'].view(B, hc * wc)
        
        # location loss
        loc_loss = l2_dist_loc_min[dist_norm_valid_mask].mean()
        
        # desc Head Loss, per-pixel level triplet loss from https://arxiv.org/pdf/1902.11046.pdf.
        desc_loss, _ = build_descriptor_loss(data['source_desc'], data['target_desc_warped'], data['target_coord_warped'].detach(), top_kk=data['border_mask'], relax_field=8)
        
        # score loss
        target_score_associated = data['target_score'].view(B, hc * wc).gather(1, l2_dist_loc_min_index).view(B, hc, wc).unsqueeze(1)
        dist_norm_valid_mask = dist_norm_valid_mask.view(B, hc, wc).unsqueeze(1) & data['border_mask'].unsqueeze(1) 
        l2_dist_loc_min = l2_dist_loc_min.view(B, hc, wc).unsqueeze(1)
        loc_err = l2_dist_loc_min[dist_norm_valid_mask]
        
        # repeatable_constrain in score loss
        repeatable_constrain = ((target_score_associated[dist_norm_valid_mask] + data['source_score'][dist_norm_valid_mask]) * (loc_err - loc_err.mean())).mean()

        # consistent_constrain in score_loss
        consistent_constrain = torch.nn.functional.mse_loss(data['target_score_warped'][data['border_mask'].unsqueeze(1)], data['source_score'][data['border_mask'].unsqueeze(1)]).mean() * 2
        aware_consistent_loss = torch.nn.functional.mse_loss(data['target_aware_warped'][data['border_mask'].unsqueeze(1).repeat(1, 2, 1, 1)], data['source_aware'][data['border_mask'].unsqueeze(1).repeat(1, 2, 1, 1)]).mean() * 2
        
        score_loss = repeatable_constrain + consistent_constrain + aware_consistent_loss
        
        loss = self.loc_weight * loc_loss + self.desc_weight * desc_loss + self.score_weight * score_loss + self.corres_weight * corres_loss
        
        return loss, self.loc_weight * loc_loss, self.desc_weight * desc_loss, self.score_weight * score_loss, self.corres_weight * corres_loss

        


