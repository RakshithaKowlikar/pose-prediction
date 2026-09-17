from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import smplx
from scipy.spatial.transform import Rotation as R, Slerp

IN_DIR = Path("data/raw")
OUT_DIR = Path("data/AMASS/npz")

TARGET_FPS = 25
JOINTS_TO_KEEP = 22
DEVICE = torch.device("cpu")
MODEL_DIR = Path("smplh1/smplh")

SMPL_MODELS = {}

MAX_BATCH_SIZE = 512

R_Z_TO_Y = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=np.float32)


def resample_motion(pose_body, trans, src_fps, tgt_fps):
    T = pose_body.shape[0]
    if T < 2 or src_fps == tgt_fps:
        return pose_body, trans

    if src_fps % tgt_fps == 0:
        step = src_fps // tgt_fps
        return pose_body[::step], trans[::step]

    src_t = np.arange(T, dtype=np.float64) / src_fps
    tgt_t = np.arange(0.0, src_t[-1] + 1e-9, 1.0 / tgt_fps)
    tgt_t = tgt_t[tgt_t <= src_t[-1]]
    if tgt_t.size < 2:
        return pose_body, trans

    trans_out = np.stack(
        [np.interp(tgt_t, src_t, trans[:, i]) for i in range(trans.shape[1])],
        axis=1,
    )

    n_joints = pose_body.shape[1] // 3
    pose_out = np.empty((tgt_t.size, pose_body.shape[1]), dtype=np.float64)
    for j in range(n_joints):
        sl = slice(3 * j, 3 * j + 3)
        rots = R.from_rotvec(pose_body[:, sl].astype(np.float64))
        pose_out[:, sl] = Slerp(src_t, rots)(tgt_t).as_rotvec()

    return pose_out.astype(np.float32), trans_out.astype(np.float32)


def axis_angle_to_rot_mat(axis_angle):
    if axis_angle.ndim != 2:
        raise ValueError(f"Unexpected axis_angle shape: {axis_angle.shape}")

    T, dim = axis_angle.shape
    rotmats = R.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    if dim > 3:
        rotmats = rotmats.reshape(T, dim // 3, 3, 3)

    return rotmats.astype(np.float32)


def rotate_z_to_y_rotmat(rotmat):
    T, J = rotmat.shape[:2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    
    rotated = R_Z_TO_Y @ rotmat_flat @ R_Z_TO_Y.T
    
    return rotated.reshape(T, J, 3, 3)


def load_smplh_model(gender: str):
    gender = gender.lower()
    if gender.startswith("m"):
        gender = "male"
    elif gender.startswith("f"):
        gender = "female"
    else:
        gender = "male"
    
    if gender not in SMPL_MODELS:
        model_file = MODEL_DIR / f"SMPLH_{gender.upper()}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing SMPL-H model file: {model_file}")
        
        SMPL_MODELS[gender] = smplx.create(
            MODEL_DIR.parent,
            model_type="smplh",
            gender=gender,
            use_pca=False,
            batch_size=MAX_BATCH_SIZE
        ).to(DEVICE)
    
    return SMPL_MODELS[gender]


def process_smpl_chunked(model, trans, global_orient, body_pose, betas, chunk_size=MAX_BATCH_SIZE):
    T = trans.shape[0]
    all_joints = []
    
    betas_expanded = betas.expand(T, -1)
    
    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)
        chunk_T = end_idx - start_idx
        
        trans_chunk = trans[start_idx:end_idx]
        global_orient_chunk = global_orient[start_idx:end_idx]
        body_pose_chunk = body_pose[start_idx:end_idx]
        betas_chunk = betas_expanded[start_idx:end_idx]
        
        left_hand_pose = torch.zeros(chunk_T, 45, device=DEVICE)
        right_hand_pose = torch.zeros(chunk_T, 45, device=DEVICE)
        
        with torch.no_grad():
            output = model(
                transl=trans_chunk,
                global_orient=global_orient_chunk,
                body_pose=body_pose_chunk,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas_chunk
            )
            joints_chunk = output.joints[:, :JOINTS_TO_KEEP, :]
            all_joints.append(joints_chunk)
    
    return torch.cat(all_joints, dim=0)


def process_file(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        gender = data["gender"].item().lower() if "gender" in data else "male"
        fps = int(data["mocap_framerate"].item())

        poses_np = np.asarray(data["poses"], dtype=np.float32)
        trans_np = np.asarray(data["trans"], dtype=np.float32)

        T_src = min(poses_np.shape[0], trans_np.shape[0])
        pose_body_np = poses_np[:T_src, :66]
        trans_np = trans_np[:T_src]

        pose_body_np, trans_np = resample_motion(
            pose_body_np, trans_np, fps, TARGET_FPS
        )

        pose_body = torch.from_numpy(np.ascontiguousarray(pose_body_np)).float()
        trans = torch.from_numpy(np.ascontiguousarray(trans_np)).float()

        T = pose_body.shape[0]

        global_orient = pose_body[:, :3]
        body_pose = pose_body[:, 3:]

        global_orient_rotmat = axis_angle_to_rot_mat(
            global_orient.cpu().numpy()
        )  
        
        body_pose_rotmat = axis_angle_to_rot_mat(
            body_pose.cpu().numpy()
        )  
        
        joint_rotmats = np.concatenate([
            global_orient_rotmat[:, None, :, :],  
            body_pose_rotmat                      
        ], axis=1)
        
        joint_rotmats = rotate_z_to_y_rotmat(joint_rotmats)
        
        joint_rotmats_flat = joint_rotmats.reshape(T, JOINTS_TO_KEEP, 9)

        model = load_smplh_model(gender)
        num_betas = model.num_betas if hasattr(model, "num_betas") else 10

        betas_np = data["betas"] if "betas" in data else np.zeros(num_betas, dtype=np.float32)
        betas_np = np.asarray(betas_np, dtype=np.float32)
        if betas_np.shape[0] > num_betas:
            betas_np = betas_np[:num_betas]
        elif betas_np.shape[0] < num_betas:
            betas_np = np.pad(betas_np, (0, num_betas - betas_np.shape[0]))
        
        betas = torch.from_numpy(betas_np).float().unsqueeze(0).to(DEVICE)
        trans = trans.to(DEVICE)
        global_orient = global_orient.to(DEVICE)
        body_pose = body_pose.to(DEVICE)
        
        joints = process_smpl_chunked(
            model, trans, global_orient, body_pose, betas, chunk_size=MAX_BATCH_SIZE
        )
        joints = joints.cpu().numpy()

        joints = joints @ R_Z_TO_Y.T
        root_pos = joints[:, 0, :]

        with torch.no_grad():
            rest_out = model(
                transl=torch.zeros(1, 3, device=DEVICE),
                global_orient=torch.zeros(1, 3, device=DEVICE),
                body_pose=torch.zeros(1, 63, device=DEVICE),
                left_hand_pose=torch.zeros(1, 45, device=DEVICE),
                right_hand_pose=torch.zeros(1, 45, device=DEVICE),
                betas=betas,
            )
        rest_joints = rest_out.joints[0, :JOINTS_TO_KEEP, :].cpu().numpy() @ R_Z_TO_Y.T

        return {
            "poses": joint_rotmats_flat,
            "trans": root_pos,
            "rest_joints": rest_joints.astype(np.float32),
            "betas": betas_np,
            "gender": gender,
            "fps": np.int32(TARGET_FPS),
            "src_fps": np.int32(fps),
            "subset": npz_path.parents[1].name,
            "subject_id": npz_path.parents[0].name,
            "sequence_id": npz_path.stem.replace("_poses", ""),
        }

    except Exception as e:
        print(f"Skipping {npz_path.name}: {e}")
        return None


def save_npz(out_path: Path, data_dict: dict):
    np.savez_compressed(out_path, **data_dict)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pose_files = list(IN_DIR.rglob("*_poses.npz"))
    print(f"Total: {len(pose_files)} pose files")

    written = {}
    collisions = 0
    skipped = 0

    for path in tqdm(pose_files):
        result = process_file(path)
        if result is None:
            skipped += 1
            continue

        out_name = (
            f"{result['subset']}_{result['subject_id']}_{result['sequence_id']}.npz"
        )
        if out_name in written:
            collisions += 1
            print(f"COLLISION: {out_name} already written by {written[out_name]}")
        written[out_name] = str(path)

        save_npz(OUT_DIR / out_name, result)

    print(f"\nWrote {len(written)} files to {OUT_DIR}")
    print(f"Skipped (unreadable): {skipped}")
    print(f"Name collisions:      {collisions}")


if __name__ == "__main__":
    main()
