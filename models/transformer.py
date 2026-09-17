import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rot6d_to_rotmat(rot6d):
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def rotmat_to_rot6d(mat):
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def project_rot6d(rot6d):
    return rotmat_to_rot6d(rot6d_to_rotmat(rot6d))


def build_skeleton_mask(nj):
    mask = torch.full((nj, nj), float("-inf"))
    edges = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 2), (2, 5), (5, 8), (8, 11),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    for i, j in edges:
        if i < nj and j < nj:
            mask[i, j] = mask[j, i] = 0.0
    mask.fill_diagonal_(0.0)
    return mask


class PosEncoding(nn.Module):
    def __init__(self, dim, maxlen=5000):
        super().__init__()
        pe = torch.zeros(maxlen, dim)
        pos = torch.arange(maxlen, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.shape[1]
        pos_enc = self.pe[:, :seq_len]
        if x.dim() == 4:
            pos_enc = pos_enc.unsqueeze(2)
        return x + pos_enc


class TemporalAttn(nn.Module):
    def __init__(self, dim, nhead, nj, dropout=0.1):
        super().__init__()
        self.nj = nj
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        self.joint_q_projs = nn.Parameter(torch.randn(nj, dim, dim))
        self.joint_k_projs = nn.Parameter(torch.randn(nj, dim, dim))
        self.joint_v_projs = nn.Parameter(torch.randn(nj, dim, dim))
        nn.init.xavier_uniform_(self.joint_q_projs)
        nn.init.xavier_uniform_(self.joint_k_projs)
        nn.init.xavier_uniform_(self.joint_v_projs)
        
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        bs, seqlen, nj, dim = x.shape
        nh, hd = self.nhead, self.head_dim
        
        q = torch.einsum('btjd,jde->btje', x, self.joint_q_projs)
        k = torch.einsum('btjd,jde->btje', x, self.joint_k_projs)
        v = torch.einsum('btjd,jde->btje', x, self.joint_v_projs)
        
        q = q.view(bs, seqlen, nj, nh, hd).permute(0, 2, 3, 1, 4).reshape(bs * nj * nh, seqlen, hd)
        k = k.view(bs, seqlen, nj, nh, hd).permute(0, 2, 3, 1, 4).reshape(bs * nj * nh, seqlen, hd)
        v = v.view(bs, seqlen, nj, nh, hd).permute(0, 2, 3, 1, 4).reshape(bs * nj * nh, seqlen, hd)
        
        scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
        attn = self.drop(F.softmax(scores, dim=-1))
        out = attn @ v
        
        out = out.view(bs, nj, nh, seqlen, hd).permute(0, 3, 1, 2, 4).reshape(bs, seqlen, nj, dim)
        return self.drop(self.out_proj(out))


class SpatialAttn(nn.Module):
    def __init__(self, dim, nhead, nj, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        self.query_projs = nn.Parameter(torch.randn(nj, dim, dim))
        nn.init.xavier_uniform_(self.query_projs)
        
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        
        self.register_buffer("attn_mask", build_skeleton_mask(nj))

    def forward(self, x):
        B, T, J, D = x.shape
        nh, hd = self.nhead, self.head_dim
        
        x_flat = x.reshape(B * T, J, D)
        
        k = self.key_proj(x_flat).view(B * T, J, nh, hd).transpose(1, 2)
        v = self.val_proj(x_flat).view(B * T, J, nh, hd).transpose(1, 2)
        q = torch.einsum('bjd,jde->bje', x_flat, self.query_projs).view(B * T, J, nh, hd).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        scores = scores + self.attn_mask.unsqueeze(0).unsqueeze(0)
        attn = self.drop(F.softmax(scores, dim=-1))
        
        out = (attn @ v).transpose(1, 2).reshape(B * T, J, D)
        return self.drop(self.out_proj(out)).view(B, T, J, D)


class STTransformerBlock(nn.Module):
    def __init__(self, dim, nhead=4, dff=256, dropout=0.1, nj=22, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        
        self.temp_attn = TemporalAttn(dim, nhead, nj, dropout)
        self.spat_attn = SpatialAttn(dim, nhead, nj, dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def _fwd(self, x):
        temp_out = self.temp_attn(x)
        spat_out = self.spat_attn(x)
        x = self.norm1(x + temp_out + spat_out)
        
        B, T, J, D = x.shape
        x_flat = x.reshape(B * T * J, D)
        x_flat = self.norm2(x_flat + self.mlp(x_flat))
        return x_flat.reshape(B, T, J, D)

    def forward(self, x):
        if self.grad_ckpt and self.training:
            return checkpoint(self._fwd, x, use_reentrant=False)
        return self._fwd(x)


class STTransformer(nn.Module):
    def __init__(self, in_frames=50, nj=22, dim=128, nlayers=8, 
                 nhead=8, dff=256, dropout=0.1, grad_ckpt=True):
        super().__init__()
        self.in_frames = in_frames
        self.nj = nj
        
        self.joint_embed_w = nn.Parameter(torch.randn(nj, 6, dim))
        self.joint_embed_b = nn.Parameter(torch.zeros(nj, dim))
        nn.init.xavier_uniform_(self.joint_embed_w)
        self.root_embed = nn.Linear(3, dim)
        
        self.pos_enc = PosEncoding(dim, maxlen=5000)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            STTransformerBlock(dim, nhead, dff, dropout, nj, grad_ckpt)
            for _ in range(nlayers)
        ])

        self.joint_head_w1 = nn.Parameter(torch.randn(nj, dim, dff))
        self.joint_head_b1 = nn.Parameter(torch.zeros(nj, dff))
        self.joint_head_w2 = nn.Parameter(torch.randn(nj, dff, 6))
        self.joint_head_b2 = nn.Parameter(torch.zeros(nj, 6))
        nn.init.xavier_uniform_(self.joint_head_w1)
        nn.init.xavier_uniform_(self.joint_head_w2)
        
        self.root_head = nn.Sequential(
            nn.Linear(dim, dff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff // 2, 3),
        )

    def embed(self, pose, root):
        pose_emb = torch.einsum('btjc,jcd->btjd', pose, self.joint_embed_w) + self.joint_embed_b
        root_emb = self.root_embed(root).unsqueeze(2)
        emb = pose_emb + root_emb
        return self.drop(self.pos_enc(emb))

    def encode(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def predict_next(self, pose, root):
        x = self.encode(self.embed(pose, root))
        h = x[:, -1]  
        
        hidden = torch.einsum('bjd,jdf->bjf', h, self.joint_head_w1) + self.joint_head_b1
        hidden = F.relu(hidden)
        pose_delta = torch.einsum('bjf,jfc->bjc', hidden, self.joint_head_w2) + self.joint_head_b2
        next_pose = project_rot6d(pose[:, -1] + pose_delta)
        
        root_delta = self.root_head(h.mean(1))
        next_root = root[:, -1] + root_delta
        
        return next_pose, next_root

    def forward(self, pose, root, n_frames):
        preds_p, preds_r = [], []
        for _ in range(n_frames):
            next_p, next_r = self.predict_next(pose, root)
            preds_p.append(next_p)
            preds_r.append(next_r)
            pose = torch.cat([pose[:, 1:], next_p.unsqueeze(1)], 1)
            root = torch.cat([root[:, 1:], next_r.unsqueeze(1)], 1)
        return torch.stack(preds_p, 1), torch.stack(preds_r, 1)

    @torch.no_grad()
    def predict(self, pose, root, n_frames):
        self.eval()
        return self.forward(pose, root, n_frames)
