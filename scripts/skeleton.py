import torch

SMPL_PARENTS = [
    -1,
    0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
]

_PARENT_IDX = [max(p, 0) for p in SMPL_PARENTS]


def rest_to_offsets(rest_joints):
    offsets = rest_joints - rest_joints[..., _PARENT_IDX, :]
    offsets[..., 0, :] = 0.0
    return offsets


def forward_kinematics(rotmats, root_pos, offsets):
    J = rotmats.shape[2]

    positions = [None] * J
    global_rot = [None] * J

    positions[0] = root_pos
    global_rot[0] = rotmats[:, :, 0]

    for j in range(1, J):
        p = SMPL_PARENTS[j]
        global_rot[j] = global_rot[p] @ rotmats[:, :, j]
        bone = torch.einsum("btac,bc->bta", global_rot[p], offsets[:, j])
        positions[j] = positions[p] + bone

    return torch.stack(positions, dim=2)
