import sys
from pathlib import Path
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_loader import AMASSForecastDataset, collate_fn
from scripts.skeleton import forward_kinematics, rest_to_offsets
from scripts.train_utils import get_lr_schedule, make_optimizer, describe
from models.baseline import PoseGRU
from models.transformer import rot6d_to_rotmat, rotmat_to_rot6d


CONFIG = {
    "input_frames": 25,
    "output_frames": 5,
    "num_joints": 22,
    "fps": 25,               
    "stride_train": 10,
    "stride_val": 10,

    "batch_size": 24,
    "epochs": 20,
    "lr": 3e-4,              
    "warmup_epochs": 2,
    "weight_decay": 1e-4,
    "adam_betas": (0.9, 0.98),
    "adam_eps": 1e-9,
    "grad_clip": 1.0,
    "dropout": 0.1,

    "val_every": 2,
    "val_batches": 60,
    "val_gt_root": False,

    "hidden_dim": 256,
    "num_layers": 2,
    "checkpoint_dir": "checkpoints/baseline_gru",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "workers": 2,
    "pin_mem": True,
}


def prepare_batch(batch, device):
    past_pose = batch["past_pose"].to(device)
    future_pose = batch["future_pose"].to(device)
    past_root = batch["past_root"].to(device)
    future_root = batch["future_root"].to(device)

    B, T_in, J, _ = past_pose.shape
    T_out = future_pose.shape[1]

    past_pose_6d = rotmat_to_rot6d(
        past_pose.view(B, T_in, J, 3, 3)
    )
    future_pose_6d = rotmat_to_rot6d(
        future_pose.view(B, T_out, J, 3, 3)
    )

    offsets = rest_to_offsets(batch["rest_joints"].to(device))

    return past_pose_6d, future_pose_6d, past_root, future_root, offsets

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        past_pose, future_pose, past_root, future_root, _ = prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=CONFIG["amp"]):
            pred_pose, pred_root = model(past_pose, past_root)
            loss = (
                F.mse_loss(pred_pose, future_pose)
                + 0.1 * F.mse_loss(pred_root, future_root)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mpjpe_list = []

    max_b = CONFIG["val_batches"]
    n_batches = max_b if max_b and max_b > 0 else len(loader)

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        past_pose, future_pose, past_root, future_root, offsets = prepare_batch(batch, device)
        pred_pose, pred_root = model(past_pose, past_root)

        pred_rot = rot6d_to_rotmat(pred_pose)
        gt_rot = rot6d_to_rotmat(future_pose)

        root_for_pred = future_root if CONFIG["val_gt_root"] else pred_root
        pred_pos = forward_kinematics(pred_rot, root_for_pred, offsets)
        gt_pos = forward_kinematics(gt_rot, future_root, offsets)

        err = torch.norm(pred_pos - gt_pos, dim=-1).mean(dim=(0, 2)) * 1000
        mpjpe_list.append(err.cpu().numpy())

    per_frame = np.stack(mpjpe_list).mean(axis=0)
    fps = CONFIG["fps"]

    return {
        "mpjpe": float(per_frame.mean()),
        "fde": float(per_frame[-1]),
        "per_horizon": {
            f"{int(round((t + 1) * 1000 / fps))}ms": float(per_frame[t])
            for t in range(per_frame.shape[0])
        },
    }


def main():
    device = torch.device(CONFIG["device"])
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AMASSForecastDataset(
        "data/AMASS/train.txt",
        CONFIG["input_frames"],
        CONFIG["output_frames"],
        CONFIG["stride_train"],
        normalize_orientation=True,
    )
    val_ds = AMASSForecastDataset(
        "data/AMASS/val.txt",
        CONFIG["input_frames"],
        CONFIG["output_frames"],
        CONFIG["stride_val"],
        normalize_orientation=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["workers"],
        collate_fn=collate_fn,
    )

    model = PoseGRU(
        num_joints=CONFIG["num_joints"],
        input_dim=CONFIG["num_joints"] * 6,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        output_frames=CONFIG["output_frames"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print(describe(CONFIG, "GRU baseline"))

    optimizer = make_optimizer(model, CONFIG)

    steps_per_ep = len(train_loader)
    scheduler = get_lr_schedule(
        optimizer,
        warmup=steps_per_ep * CONFIG["warmup_epochs"],
        total=steps_per_ep * CONFIG["epochs"],
    )
    scaler = torch.amp.GradScaler(enabled=CONFIG["amp"])

    best = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch)

        if epoch % CONFIG["val_every"] == 0:
            metrics = validate(model, val_loader, device)
            print(f"Epoch {epoch} | MPJPE: {metrics['mpjpe']:.2f} mm | FDE: {metrics['fde']:.2f} mm")
            print("  " + "  ".join(f"{k} {v:.1f}" for k, v in metrics["per_horizon"].items()))

            if metrics["mpjpe"] < best:
                best = metrics["mpjpe"]
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

    print(f"\nBest GRU MPJPE: {best:.2f} mm")


if __name__ == "__main__":
    main()
