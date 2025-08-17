# aerial_gym/utils/custom_metrics.py
import os
import numpy as np
import torch
import time
import json
from typing import Dict, Optional

try:
    from PIL import Image  # pip install pillow
    _PIL_OK = True
except Exception:
    _PIL_OK = False

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None



class SlamMetrics:
    """
    Lightweight TB/W&B logger for scalars, episodes, and coverage images.
    """
    def __init__(self, log_dir: str, num_envs: int, save_maps_every: int = 10, use_wandb: bool = False):
        self.num_envs = num_envs
        self.global_step = 0
        self.save_maps_every = int(save_maps_every)
        self.use_wandb = use_wandb

        self.log_dir = log_dir
        self.cov_dir = os.path.join(self.log_dir, "coverage")
        os.makedirs(self.cov_dir, exist_ok=True)
        self.scalars_path = os.path.join(self.log_dir, "scalars.jsonl")

        # tiny helper to avoid spamming errors
        self._warned_pil = False

        self.tb = None
        if SummaryWriter is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.tb = SummaryWriter(log_dir=log_dir)

        self._ep_idx = torch.zeros(num_envs, dtype=torch.long)
        self._log_dir = log_dir
        self._map_dir = os.path.join(log_dir, "coverage")
        os.makedirs(self._map_dir, exist_ok=True)

    def log_scalar(self, name: str, value: float, step: int):
        rec = {"t": time.time(), "step": int(step), "name": name, "value": float(value)}
        with open(self.scalars_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_dict(self, values: Dict[str, float], step: int):
        for k, v in values.items():
            self.log_scalar(k, float(v), step)

    def maybe_save_coverage(self, coverage_bool, step: int, env_ids: Optional[list] = None):
        """Save a binary coverage PNG per selected env. coverage_bool: (N,H,W) or (N,A,H,W) bool."""
        if self.save_maps_every <= 0 or (step % self.save_maps_every) != 0:
            return
        if coverage_bool is None:
            return

        cov = coverage_bool
        # Reduce agents if necessary -> (N,H,W)
        if cov.dim() == 4:  # (N,A,H,W)
            cov = cov.any(dim=1)
        if cov.dtype.__str__() != "torch.bool":
            cov = cov.bool()

        cov_np = cov.detach().to("cpu").numpy().astype(np.uint8) * 255  # (N,H,W) in {0,255}
        N = cov_np.shape[0]
        if env_ids is None:
            env_ids = list(range(min(N, 4)))  # save first few by default

        if not _PIL_OK:
            if not self._warned_pil:
                print("[SlamMetrics] WARNING: pillow not available; coverage images won't be saved. `pip install pillow`")
                self._warned_pil = True
            return

        for e in env_ids:
            img = Image.fromarray(cov_np[e])
            img = img.resize((cov_np.shape[2]*3, cov_np.shape[1]*3), resample=Image.NEAREST)  # 3x scale for visibility
            out = os.path.join(self.cov_dir, f"step_{step:06d}_env_{e}.png")
            img.save(out)
    
    def _grid_to_rgb(self, grid_u8: torch.Tensor) -> torch.Tensor:
        """
        grid_u8: (H,W) with values {0,1,2}
        returns: (H,W,3) uint8 -> unseen=dark gray, free=light gray, obstacle=red
        """
        H, W = grid_u8.shape
        out = torch.zeros((H, W, 3), dtype=torch.uint8, device=grid_u8.device)
        out[..., :] = 50                   # unseen
        out[grid_u8 == 1] = torch.tensor([200, 200, 200], dtype=torch.uint8, device=grid_u8.device)  # free
        out[grid_u8 == 2] = torch.tensor([200, 40, 40],   dtype=torch.uint8, device=grid_u8.device)  # obstacle
        return out


    @staticmethod
    def _to1d(x: torch.Tensor) -> torch.Tensor:
        return x.detach().reshape(-1).float()

    def step(self, coverage_frac: torch.Tensor, collided: torch.Tensor, alt_err: torch.Tensor, action_mag: torch.Tensor):
        """
        Log per-step means across envs.
        Tensors can be shape (N,) or (N,1).
        """
        self.global_step += 1
        cov = self._to1d(coverage_frac)
        col = self._to1d(collided).float()
        alt = self._to1d(alt_err)
        amag = self._to1d(action_mag)

        if self.tb is not None:
            self.tb.add_scalar("train/coverage_mean", cov.mean().item(), self.global_step)
            self.tb.add_scalar("train/collision_rate", col.mean().item(), self.global_step)
            self.tb.add_scalar("train/alt_error_mean", alt.mean().item(), self.global_step)
            self.tb.add_scalar("train/action_mag_mean", amag.mean().item(), self.global_step)

        if self.use_wandb:
            import wandb
            wandb.log({
                "train/coverage_mean": cov.mean().item(),
                "train/collision_rate": col.mean().item(),
                "train/alt_error_mean": alt.mean().item(),
                "train/action_mag_mean": amag.mean().item(),
                "global_step": self.global_step
            })

    def end_episode(self, env_ids: torch.Tensor, ep_return: torch.Tensor, ep_len: torch.Tensor,
                    final_coverage: torch.Tensor, ep_collisions: torch.Tensor,
                    coverage_grid: torch.Tensor = None):
        """
        Log per-episode stats for each env_id in env_ids.
        Also optionally log/save a coverage image for those envs.
        """
        env_ids = env_ids.detach().cpu().long().view(-1)
        R = self._to1d(ep_return).cpu().numpy()
        L = self._to1d(ep_len).cpu().numpy()
        C = self._to1d(final_coverage).cpu().numpy()
        K = self._to1d(ep_collisions).cpu().numpy()

        for i, env_id in enumerate(env_ids.tolist()):
            self._ep_idx[env_id] += 1
            ep_i = int(self._ep_idx[env_id].item())

            if self.tb is not None:
                self.tb.add_scalar(f"episode/return_env{env_id}", float(R[i]), ep_i)
                self.tb.add_scalar(f"episode/len_env{env_id}",    float(L[i]), ep_i)
                self.tb.add_scalar(f"episode/final_coverage_env{env_id}", float(C[i]), ep_i)
                self.tb.add_scalar(f"episode/collisions_env{env_id}",     float(K[i]), ep_i)

            if self.use_wandb:
                import wandb
                wandb.log({
                    f"episode/return_env{env_id}": float(R[i]),
                    f"episode/len_env{env_id}": float(L[i]),
                    f"episode/final_coverage_env{env_id}": float(C[i]),
                    f"episode/collisions_env{env_id}": float(K[i]),
                    "episode_index": ep_i
                })

            # optional map snapshot
            if coverage_grid is not None and ep_i % self.save_maps_every == 0:
                try:
                    img = self.coverage_to_rgb(coverage_grid[env_id])  # (H,W,3) uint8
                    path = os.path.join(self._map_dir, f"env{env_id:02d}_ep{ep_i:05d}.png")
                    import imageio
                    imageio.imwrite(path, img)
                    if self.tb is not None:
                        self.tb.add_image(f"coverage/env{env_id}", img.transpose(2,0,1), ep_i)
                except Exception:
                    pass

    @staticmethod
    def coverage_to_rgb(grid: torch.Tensor) -> np.ndarray:
        """
        Convert a float tensor (H,W) in [0,1] or a bool mask to an RGB image.
        """
        g = grid.detach().float().clamp(0, 1).cpu().numpy()
        g = (g * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
