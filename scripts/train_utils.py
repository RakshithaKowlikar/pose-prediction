import math
import torch

def get_lr_schedule(opt, warmup, total):
    def lr_fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)


def make_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["adam_betas"],
        eps=cfg["adam_eps"],
    )


def describe(cfg, model_name):
    return (
        f"{model_name} | {cfg['input_frames']}->{cfg['output_frames']} frames @ "
        f"{cfg['fps']}Hz | bs={cfg['batch_size']} lr={cfg['lr']:g} "
        f"epochs={cfg['epochs']} wd={cfg['weight_decay']:g} clip={cfg['grad_clip']}"
    )
