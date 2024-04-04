#package_path = './PPConv/pvcnn_code'
#sys.path.append(package_path)

import torch
from module.projection import Projection
from module.layers import AttentionStack, AttentionStackConfig,AttentionBlockConfig
from module.tpv_aggregator import TPVAggregator, TPVDecoder

'''TEST'''
print('Starting...')
print("PyTorch version:", torch.__version__)
print(torch.cuda.is_available())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Input Simulation'''
B = 2
Np = 1024
C = 9
input_pc = torch.randn((B, C + 3, Np))
input_pc = input_pc.to(device)
print(input_pc.shape)

'''Forward Testing'''


'''1: Point to Feature Plane Projection'''
'''1.1 Voxelization through scatter function'''
features = input_pc[:, :-3, :]
coords = input_pc[:, -3:, :]
B, C, Np = features.shape
R = 32  # resolution
eps = 1e-8

dev = features.get_device()
norm_coords = coords - coords.mean(dim=2, keepdim=True)
norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True)[0] * 2.0 + eps) + 0.5
norm_coords = torch.clamp(norm_coords * (R - 1), 0, R - 1 - eps)

sample_idx = torch.arange(B, dtype=torch.int64, device=dev)
sample_idx = sample_idx.unsqueeze(-1).expand(-1, Np).reshape(-1).unsqueeze(1)
norm_coords = norm_coords.transpose(1, 2).reshape(B * Np, 3)
coords_int = torch.round(norm_coords).to(torch.int64)
coords_int = torch.cat((sample_idx, coords_int), 1)
p_v_dist = torch.cat((sample_idx, torch.abs(norm_coords - coords_int[:, 1:])), 1)

'''1.2 Triplane Projection'''
# TODO: When R is different in triplane
in_channels = C
mid_channels = 128
projection = Projection(R, in_channels, mid_channels, eps=eps).to(device)
proj_axes = [1,2,3]
proj_feat = []

'''TODO: idividual feature transform'''
if 1 in proj_axes:
    proj_x = projection(features, norm_coords, coords_int, p_v_dist, 1).permute(0, 3, 1, 2)
    proj_feat.append(proj_x)
if 2 in proj_axes:
    proj_y = projection(features, norm_coords, coords_int, p_v_dist, 2).permute(0, 3, 1, 2)
    proj_feat.append(proj_y)
if 3 in proj_axes:
    proj_z = projection(features, norm_coords, coords_int, p_v_dist, 3).permute(0, 3, 1, 2)
    proj_feat.append(proj_z)
#print(proj_x.shape)  # B, Cproj, R, R
proj_feat = torch.stack(proj_feat,-1)
print(proj_feat.shape)

'''2:Plane to Vector Flatten'''
'''Optional: Spatial aware flatten'''
'''2.1 Flatten: [B,Cproj,R,R,3] -> [B,Cproj,R*R*3]'''  # Ordering Important!!
B,C_proj,R,R,pl = proj_feat.shape
proj_feat_flatten = proj_feat.reshape(B,C_proj,-1)
proj_feat_flatten = proj_feat_flatten.transpose(1, 2).contiguous() #( B, C,L) -> (B, L, C)

print(proj_feat_flatten.shape)

'''3.Sequence to Sequence Modelling'''
'''3.1 Transformer'''
attentionblock_config = AttentionBlockConfig(embed_dim=128)
attentionstack_config = AttentionStackConfig(block=attentionblock_config)
S2S_model = AttentionStack(attentionstack_config).to(device)
proj_feat_flatten_transformed = S2S_model(proj_feat_flatten)
print(proj_feat_flatten_transformed.shape)

'''3.2 TODO: Mamba'''
'''3.3 reshape to triplane'''
triplane_feat = proj_feat_flatten_transformed.reshape(B,R*R,3,mid_channels)
triplane_feat = triplane_feat.permute(0,1,3,2).contiguous()
print(triplane_feat.shape)
tpv_list = [triplane_feat[...,0],triplane_feat[...,1],triplane_feat[...,2]]

'''4.Triplane Feature Gathering'''
TPVagg = TPVAggregator(R,R,R).to(device)
tpv_feat_pts = TPVagg(tpv_list,norm_coords.reshape(B,Np,3))

print(tpv_feat_pts.shape)
'''Optional: Point branch'''

'''5. TPV feature decoder'''
decoder = TPVDecoder(in_dims=128,nbr_classes=10).to(device)
point_logits = decoder(tpv_feat_pts)
print(point_logits.shape)


