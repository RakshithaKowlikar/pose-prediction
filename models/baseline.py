import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import project_rot6d


class PoseGRU(nn.Module):

    def __init__(
        self,
        num_joints=22,
        input_dim=132,        
        hidden_dim=256,
        num_layers=2,
        output_frames=5,
        dropout=0.1,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.output_frames = output_frames

        self.root_encoder = nn.Linear(3, 32)

        self.gru = nn.GRU(
            input_size=input_dim + 32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_frames * input_dim),
        )

        self.root_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_frames * 3),
        )

    def forward(self, past_pose_6d, past_root):
        B, T, J, _ = past_pose_6d.shape

        pose_flat = past_pose_6d.reshape(B, T, -1) 

        root_enc = self.root_encoder(past_root)    

        x = torch.cat([pose_flat, root_enc], dim=-1)

        _, h_n = self.gru(x)
        h = h_n[-1] 

        future_pose = self.pose_decoder(h)
        future_pose = future_pose.view(
            B, self.output_frames, J, 6
        )
        future_pose = project_rot6d(future_pose)

        future_root = self.root_decoder(h)
        future_root = future_root.view(
            B, self.output_frames, 3
        )

        return future_pose, future_root
