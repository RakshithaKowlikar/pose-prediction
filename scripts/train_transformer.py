import sys
import time
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
from models.transformer import STTransformer, rot6d_to_rotmat, rotmat_to_rot6d

fk = forward_kinematics

cfg = {
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

    "d_model": 128,
    "num_layers": 8,
    "num_heads": 8,
    "dim_feedforward": 256,

    "sched_sampling": True,
    "ss_start": 6,
    "ss_ramp": 8,
    "ss_min": 0.1,

    "grad_ckpt": True,

    "ckpt_dir": "checkpoints/st_transformer",
    "save_every": 10,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "workers": 2,
    "pin_mem": True,
}


def prep_batch(batch, dev):
    past_p = batch["past_pose"].to(dev, non_blocking=True)
    fut_p = batch["future_pose"].to(dev, non_blocking=True)
    past_r = batch["past_root"].to(dev, non_blocking=True)
    fut_r = batch["future_root"].to(dev, non_blocking=True)
    
    B, T_in, J = past_p.shape[:3]
    T_out = fut_p.shape[1]
    
    past_3x3 = past_p.view(B, T_in, J, 3, 3)
    fut_3x3 = fut_p.view(B, T_out, J, 3, 3)
    
    past_6d = rotmat_to_rot6d(past_3x3)
    fut_6d = rotmat_to_rot6d(fut_3x3)

    offsets = rest_to_offsets(batch["rest_joints"].to(dev, non_blocking=True))

    return past_6d, fut_6d, past_r, fut_r, past_3x3, fut_3x3, offsets


def train_epoch(model, loader, opt, sched, scaler, dev, ep):
    model.train()
    total = 0.0
    
    if cfg["sched_sampling"] and ep >= cfg["ss_start"]:
        prog = min(1.0, (ep - cfg["ss_start"]) / cfg["ss_ramp"])
        tf_prob = max(cfg["ss_min"], 1.0 - prog)
    else:
        tf_prob = 1.0

    pbar = tqdm(loader, desc=f"Epoch {ep} [TF={tf_prob:.2f}]")
    for batch in pbar:
        past_6d, fut_6d, past_r, fut_r, _, _, _ = prep_batch(batch, dev)
        
        B, T_in, J = past_6d.shape[:3]
        T_out = fut_6d.shape[1]
        
        opt.zero_grad(set_to_none=True)
        
        losses = []
        inp_p = past_6d.clone()
        inp_r = past_r.clone()
        
        with torch.amp.autocast("cuda", enabled=cfg["amp"]):
            for t in range(T_out):
                pred_p, pred_r = model.predict_next(inp_p, inp_r)
                
                tgt_p = fut_6d[:, t, :, :]
                tgt_r = fut_r[:, t, :]
                
                l_p = F.mse_loss(pred_p, tgt_p)
                l_r = F.mse_loss(pred_r, tgt_r)
                losses.append(l_p + 0.1 * l_r)
                
                use_gt = torch.rand(B, device=dev) < tf_prob 
                
                next_p = torch.where(use_gt[:, None, None], tgt_p, pred_p.detach())
                next_r = torch.where(use_gt[:, None], tgt_r, pred_r.detach())
                
                inp_p = torch.cat([inp_p[:, 1:], next_p.unsqueeze(1)], dim=1)
                inp_r = torch.cat([inp_r[:, 1:], next_r.unsqueeze(1)], dim=1)
            
            loss = torch.stack(losses).mean()
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        
        if sched:
            sched.step()
        
        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}")
    
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, dev, max_b=20):
    model.eval()
    mpjpe_vals, fde_vals = [], []
    
    gt_root = cfg["val_gt_root"]
    metric = "Pose-only MPJPE" if gt_root else "Full motion MPJPE"
    
    n_batches = max_b if max_b and max_b > 0 else len(loader)

    for i, batch in enumerate(tqdm(loader, desc=f"Val [{metric}]", total=n_batches)):
        if i >= n_batches:
            break

        past_6d, fut_6d, past_r, fut_r, _, fut_3x3, offsets = prep_batch(batch, dev)
        T_out = fut_6d.shape[1]

        pred_6d, pred_r = model.forward(past_6d, past_r, T_out)
        pred_3x3 = rot6d_to_rotmat(pred_6d)

        root_for_pred = fut_r if gt_root else pred_r
        pred_pos = fk(pred_3x3, root_for_pred, offsets)
        gt_pos = fk(fut_3x3, fut_r, offsets)

        err = torch.norm(pred_pos - gt_pos, dim=-1).mean(dim=(0, 2)) * 1000
        mpjpe_vals.append(err.cpu().numpy())

    per_frame = np.stack(mpjpe_vals).mean(axis=0)
    fps = cfg["fps"]

    return {
        "mpjpe": float(per_frame.mean()),
        "fde": float(per_frame[-1]),
        "per_horizon": {
            f"{int(round((t + 1) * 1000 / fps))}ms": float(per_frame[t])
            for t in range(per_frame.shape[0])
        },
        "type": metric,
    }


def main():
    dev = torch.device(cfg["device"])
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = AMASSForecastDataset(
        split_file="data/AMASS/train.txt",
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride_train"],
        normalize_orientation=True,
    )
    val_ds = AMASSForecastDataset(
        split_file="data/AMASS/val.txt",
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride_val"],
        normalize_orientation=True,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["workers"], pin_memory=cfg["pin_mem"],
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["workers"], pin_memory=cfg["pin_mem"],
        collate_fn=collate_fn, drop_last=False,
    )
    
    model = STTransformer(
        in_frames=cfg["input_frames"],
        nj=cfg["num_joints"],
        dim=cfg["d_model"],
        nlayers=cfg["num_layers"],
        nhead=cfg["num_heads"],
        dff=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        grad_ckpt=cfg["grad_ckpt"],
    ).to(dev)
    
    print(describe(cfg, "ST-Transformer"))

    opt = make_optimizer(model, cfg)
    
    steps_per_ep = len(train_loader)
    total_steps = steps_per_ep * cfg["epochs"]
    warmup = steps_per_ep * cfg["warmup_epochs"]
    
    sched = get_lr_schedule(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])
    
    best = float("inf")
    t0 = time.time()
    
    for ep in range(1, cfg["epochs"] + 1):
        t_ep = time.time()
        
        tr_loss = train_epoch(model, train_loader, opt, sched, scaler, dev, ep)
        
        elapsed = time.time() - t_ep
        total_t = time.time() - t0
        eta = (cfg["epochs"] - ep) * (elapsed / 60.0)
        
        print(f"\nEpoch {ep}/{cfg['epochs']} | loss={tr_loss:.4f}")
        print(f"Time: {elapsed/60:.1f}m | Total: {total_t/3600:.2f}h | ETA: {eta:.0f}m")
        
        if ep % cfg["val_every"] == 0 or ep == cfg["epochs"]:
            metrics = validate(model, val_loader, dev, cfg["val_batches"])
            print(f"{metrics['type']}: {metrics['mpjpe']:.2f}mm | FDE: {metrics['fde']:.2f}mm")
            print("  " + "  ".join(f"{k} {v:.1f}" for k, v in metrics["per_horizon"].items()))
            
            if metrics["mpjpe"] < best:
                best = metrics["mpjpe"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": cfg,
                }, ckpt_dir / "best_model.pth")
                print("Saved best_model.pth")
        
        if ep % cfg["save_every"] == 0:
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": cfg,
            }, ckpt_dir / f"checkpoint_epoch_{ep}.pth")
            print(f"Saved checkpoint_epoch_{ep}.pth") 
    
    print(f"\nBest {metrics['type']}: {best:.2f}mm")
    print(f"Checkpoints: {ckpt_dir}")

if __name__ == "__main__":
    main()
