import torch
import torch.nn as nn
from module.projection import Projection
from module.layers import AttentionStack, AttentionStackConfig, AttentionBlockConfig
from module.tpv_aggregator import TPVAggregator, TPVDecoder

class Trimamba(nn.Module):
    def __init__(self, R=32, in_channels=9, mid_channels=128, embed_dim=128, nbr_classes=10, eps=1e-8):
        super(Trimamba, self).__init__()
        self.R = R
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.nbr_classes = nbr_classes
        self.eps = eps
        self.attentionblock_config = AttentionBlockConfig(embed_dim=embed_dim)
        self.attentionstack_config = AttentionStackConfig(block=self.attentionblock_config,n_layer=4)

        self.projection = Projection(R, in_channels, mid_channels, eps=eps)
        self.S2S_model = AttentionStack(self.attentionstack_config)
        self.TPVagg = TPVAggregator(R, R, R)
        self.decoder = TPVDecoder(in_dims=embed_dim, nbr_classes=nbr_classes)

    def forward(self, input_pc):
        B, _, Np = input_pc.shape
        features = input_pc[:, -3:, :]
        coords = input_pc[:, :-3, :]

        dev = features.device
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + self.eps) + 0.5
        norm_coords = torch.clamp(norm_coords * (self.R - 1), 0, self.R - 1 - self.eps)

        sample_idx = torch.arange(B, dtype=torch.int64, device=dev)
        sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)
        norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
        coords_int = torch.round(norm_coords).to(torch.int64)
        coords_int = torch.cat((sample_idx, coords_int), 1)
        p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords - coords_int[:, 1:])), 1)

        proj_axes = [1, 2, 3]
        proj_feat = []

        if 1 in proj_axes:
            proj_x = self.projection(features, norm_coords, coords_int, p_v_dist, 1).permute(0, 3, 1, 2)
            proj_feat.append(proj_x)
        if 2 in proj_axes:
            proj_y = self.projection(features, norm_coords, coords_int, p_v_dist, 2).permute(0, 3, 1, 2)
            proj_feat.append(proj_y)
        if 3 in proj_axes:
            proj_z = self.projection(features, norm_coords, coords_int, p_v_dist, 3).permute(0, 3, 1, 2)
            proj_feat.append(proj_z)
        proj_feat = torch.stack(proj_feat, -1)

        B, C_proj, R, _, _ = proj_feat.shape
        proj_feat_flatten = proj_feat.reshape(B, C_proj, -1)
        proj_feat_flatten = proj_feat_flatten.transpose(1, 2).contiguous()

        proj_feat_flatten_transformed = self.S2S_model(proj_feat_flatten)

        triplane_feat = proj_feat_flatten_transformed.reshape(B, R * R, 3, self.mid_channels)
        triplane_feat = triplane_feat.permute(0, 1, 3, 2).contiguous()
        tpv_list = [triplane_feat[..., 0], triplane_feat[..., 1], triplane_feat[..., 2]]

        tpv_feat_pts = self.TPVagg(tpv_list, norm_coords.reshape(B, Np, 3)) #B,Np,C_proj

        # for classification:
        tpv_feat_global,_ = torch.max(tpv_feat_pts, dim=1) # B,C_proj
        shape_logits = self.decoder(tpv_feat_global)

        return shape_logits


# Test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_pc = torch.randn((2, 12, 1024)).to(device)  # Assuming input features have 12 channels
    model = Trimamba().to(device)
    output = model(input_pc)
    print("Output shape:", output.shape)
