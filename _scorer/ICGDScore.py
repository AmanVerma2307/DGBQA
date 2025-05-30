import torch

class icgdScore(torch.nn.Module):

    def __init__(self,G,I):
        super().__init__()
        self.G = G # Total number of gestures
        self.I = I # Total number of Identities

    def get_HGRmask(self,y_hgr):
        return (y_hgr.unsqueeze(-1) != y_hgr).to(dtype=torch.float32)

    def get_IDmask(self,y_id):

        """
        ID mask of ICGD loss

        INPUTS:-
        1) y_id: Input ID labels of shape (B,)
        2) device: Processing device

        OUTPUTS:-
        1) id_mask: ID mask of shape (B,B). High if different samples but same identities.
        """

        ##### Similar identity mask
        id_mask = (y_id.unsqueeze(-1) == y_id).to(dtype=torch.float32)
        id_mask_curr = (y_id.unsqueeze(-1) == 1).to(dtype=torch.float32)

        ##### Distinct position mask
        B = y_id.size(0) # batch_size
        device_id = torch.device(id_mask.get_device())
        id_mask_dist = torch.logical_not(torch.eye(B).to(torch.bool)).to(device_id,dtype=torch.float32) # Except diagonal entries everything turned high
        id_mask_dist.requires_grad = False # Making this a non-parametric model.

        ##### ID mask
        return torch.logical_and(id_mask,id_mask_dist*id_mask_curr).to(dtype=torch.float32)
    
    def forward(self,y_hgr,f_theta):

        """
        ICGD Loss

        INPUTS:-
        1) y_hgr: HGR labels of shape (B,)
        2) f_theta: Output embeddings of shape (B,d)

        OUTPUTS:-
        1) icgdScoreVal: mean(G_bar*hgr_mask)
        2) icgdScoreValFull: sum(G_bar*gamma)/sum(gamma)
        """

        ##### Gram matrix formation
        f_theta = torch.nn.functional.normalize(torch.from_numpy(f_theta),dim=-1) # L2 normalization
        G_bar = torch.matmul(f_theta,torch.permute(f_theta,(1,0))) # Gram matrix -> (B,B)

        ##### HGR mask
        hgr_mask = self.get_HGRmask(torch.from_numpy(y_hgr)) # HGR mask, shape -> (B,B)

        ##### ICGD score
        G_bar = hgr_mask*hgr_mask

        ##### Negative mask
        gamma = (G_bar >= 0).to(dtype=torch.float32) # Masks negative values to zero. Shape -> (B,B)

        ##### Score values
        icgdScoreVal = torch.mean(G_bar).item()
        icgdScoreValFull = (torch.sum(G_bar*gamma)/torch.sum(gamma)).item()

        return icgdScoreVal, icgdScoreValFull
