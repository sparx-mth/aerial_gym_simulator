from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.logging import CustomLogger
from gym.spaces import Dict, Box
from typing import Dict as TypingDict

logger = CustomLogger("multiagent_slam_task")


class CustomTask(BaseTask):
    def __init__(self, task_config, **kwargs):
        super().__init__(task_config)
        self.device = self.task_config.device

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name, 
            device=self.device,
            args=self.task_config.args,
        )

        rm = getattr(self.sim_env, "robot_manager", None)
        if rm is not None and hasattr(rm, "actions"):
            self.agent_count = rm.actions.shape[0]
        else:
            self.agent_count = getattr(self.task_config, "agent_count", 3)
        self.num_envs = self.sim_env.num_envs
        # self.agent_count = self.task_config.agent_count  # expected: 3–5
        # self.coverage_maps = torch.zeros((self.num_envs, self.agent_count, 100, 100), device=self.device)

        # compute obs dim from agent_count
        obs_dim = self.agent_count * (7 + 6 + 6)  # 19 per agent
        self.task_config.observation_space_dim = obs_dim
        self.coverage_maps = torch.zeros((self.num_envs, self.agent_count, 100, 100), device=self.device)
        priv_dim = self.task_config.privileged_observation_space_dim

        self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)  # 6D continuous action

        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._warmup_len = 50                   # steps
        self._takeoff_vz = 0.8                  # m/s up during warmup

        self._dt = getattr(self.task_config, "sim_dt", 1.0/60.0)  # override in config if you know it
        self._z_ref = 1.5
        self._z_max = 5.0
        self._est_z = torch.zeros(self.num_envs, 1, device=self.device)  # env-level; or (num_envs, agent_count) if you want per-agent


        self.task_obs = {
        "observations": torch.zeros((self.num_envs, obs_dim), device=self.device),
        "privileged_obs": torch.zeros((self.num_envs, 0), device=self.device),
        "collisions": torch.zeros((self.num_envs, 1), device=self.device),
        "rewards": torch.zeros((self.num_envs, 1), device=self.device),
        }


    def close(self):
        self.sim_env.delete_env()

    def reset(self, *, seed=None, options=None):
        # optional: use seed/options if you need
        self.sim_env.reset()
        self.coverage_maps.zero_()
        self._update_observation()
        info = {}  # could add anything you want to expose
        return self.task_obs, info
    
    def reset_done(self):
        # if you track per-env terminations, reset only those here.
        # otherwise just refresh obs and return a dict like reset()
        self._update_observation()
        return self.task_obs

    def reset_idx(self, env_ids):
        self.sim_env.reset_idx(env_ids)
        self.coverage_maps[env_ids] = 0

    def render(self):
        return self.sim_env.render()

    def step(self, actions):
        # ensure torch
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(self.device).float()

        # map 6->4
        env_actions = actions[..., :4]

        # warmup override: send [vx,vy,vz,yaw] = [0,0,+vz,0] for first _warmup_len steps per env
        warm_mask = (self._steps < self._warmup_len).view(-1, 1)  # (N,1)
        if env_actions.dim() == 2 and env_actions.shape[0] == self.agent_count:
            # agent-major actions; apply warmup to all agents
            wz = torch.zeros_like(env_actions[:, 0])
            env_actions = env_actions.clone()
            env_actions[:, 0] = 0.0
            env_actions[:, 1] = 0.0
            env_actions[:, 2] = torch.where(warm_mask[0, 0], torch.full_like(wz, self._takeoff_vz), env_actions[:, 2])
            env_actions[:, 3] = 0.0
        elif env_actions.dim() == 2 and env_actions.shape[0] == self.num_envs:
            # env-major actions
            env_actions = env_actions.clone()
            env_actions[:, 0] = 0.0
            env_actions[:, 1] = 0.0
            env_actions[:, 2] = torch.where(warm_mask.squeeze(1), torch.full_like(env_actions[:, 2], self._takeoff_vz), env_actions[:, 2])
            env_actions[:, 3] = 0.0

        # clamp to reasonable bounds for the controller
        env_actions = torch.clamp(env_actions, -1.0, 1.0)

        # step sim
        self.sim_env.step(actions=env_actions)

        # step counter
        self._steps += 1

        # Update obs/reward/termination
        self._update_observation()
        self._compute_rewards()

        terminated, truncated = self._get_terminated_truncated()
        terminated = terminated.view(self.num_envs)                  # (N,)
        truncated  = truncated.view(self.num_envs)                   # (N,)

        rewards_env = self.task_obs["rewards"]
        # rewards may be (N,1) or (N,1,1); force to (N,)
        rewards_env = rewards_env.reshape(self.num_envs, -1)[:, 0]   # (N,)

        # done_mask = (terminated | truncated)
        # if done_mask.any():
        #     ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        #     self.reset_idx(ids)
        #     self._steps[ids] = 0


        return self.task_obs, rewards_env, terminated, truncated, {}


    def _update_observation(self):
        pose, vel, imu = self._extract_state_from_sim()
        # sanitize pose/vel/imu just in case
        pose = torch.nan_to_num(pose, nan=0.0, posinf=1e6, neginf=-1e6)
        vel  = torch.nan_to_num(vel,  nan=0.0, posinf=1e6, neginf=-1e6)
        imu  = torch.nan_to_num(imu,  nan=0.0, posinf=1e6, neginf=-1e6)

        flat_obs = torch.cat(
            [pose.reshape(self.num_envs, -1),
            vel.reshape(self.num_envs, -1),
            imu.reshape(self.num_envs, -1)],
            dim=-1,
        )

        # final clamp to a sane range to help the running mean/std
        flat_obs = torch.clamp(flat_obs, -1e3, 1e3)
        self.task_obs["observations"].copy_(flat_obs)
        self._update_coverage(pose)  # positions are in pose[..., :3]


    def _extract_state_from_sim(self):
        """
        Returns:
        pose: (envs, agents, 7) -> [px,py,pz,qx,qy,qz,qw]
        vel:  (envs, agents, 6) -> [vx,vy,vz, wx,wy,wz]
        imu:  (envs, agents, 6) -> [ax,ay,az, gx,gy,gz] (zeros if not available)
        """
        rm = getattr(self.sim_env, "robot_manager", None)
        if rm is None:
            raise RuntimeError("sim_env has no 'robot_manager' attribute; cannot extract states.")

        # Common Isaac/IsaacGym names to try
        candidate_names = [
            "actor_root_state_tensor",
            "root_state_tensor",
            "root_states",
            "rb_states",
            "rigid_body_states",
            "state_tensor",
            "states",
        ]

        def _try_get_tensor(obj, names):
            for n in names:
                if hasattr(obj, n):
                    t = getattr(obj, n)
                    if isinstance(t, torch.Tensor):
                        return t, n
            return None, None

        # 1) Try on robot_manager
        root_states, src_name = _try_get_tensor(rm, candidate_names)

        # 2) Try directly on sim_env if not found
        if root_states is None:
            root_states, src_name = _try_get_tensor(self.sim_env, candidate_names)

        # 3) Try per-robot list/dict
        if root_states is None:
            robots = getattr(rm, "robots", None)
            if isinstance(robots, (list, tuple)) and len(robots) > 0:
                per = []
                for i, rob in enumerate(robots):
                    t, _ = _try_get_tensor(rob, candidate_names)
                    if t is None:
                        continue
                    per.append(t)
                if per:
                    root_states = torch.stack(per, dim=1)  # (envs, agents, feat) OR (agents, feat)
                    src_name = "robots[*].<state_tensor>"
                    # If shape ends up (agents, feat), expand envs dimension
                    if root_states.dim() == 2:
                        root_states = root_states.unsqueeze(0).repeat(self.num_envs, 1, 1)

        if root_states is None:
            # One-shot debug dump (keep it)
            if not hasattr(self, "_logged_attr_dump"):
                self._logged_attr_dump = True
                print("[CustomTask] Could not find root states. Available robot_manager attrs:")
                for k in dir(rm):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(rm, k)
                        if isinstance(v, torch.Tensor):
                            print(f"  - rm.{k}: shape={tuple(v.shape)}")
                    except Exception:
                        pass
                print("[CustomTask] Available sim_env attrs:")
                for k in dir(self.sim_env):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(self.sim_env, k)
                        if isinstance(v, torch.Tensor):
                            print(f"  - sim_env.{k}: shape={tuple(v.shape)}")
                    except Exception:
                        pass

            # Fallback: fabricate zeros so training can proceed
            # (Optionally infer agent_count from rm.actions)
            if hasattr(rm, "actions") and isinstance(rm.actions, torch.Tensor):
                self.agent_count = int(rm.actions.shape[0])

            pose = torch.zeros((self.num_envs, self.agent_count, 7), device=self.device)
            pose[..., 6] = 1.0  # unit quaternion
            vel  = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)
            imu  = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)

            if not hasattr(self, "_warned_no_states"):
                self._warned_no_states = True
                print("[CustomTask] WARNING: no kinematic state found; using zeroed pose/vel/imu. Wire root states later.")

            return pose, vel, imu


        # Ensure tensor on correct device
        if not isinstance(root_states, torch.Tensor):
            root_states = torch.as_tensor(root_states, device=self.device)
        else:
            root_states = root_states.to(self.device)

        # Normalize to (envs, agents, F)
        if root_states.dim() == 2:
            # Expect (envs*agents, F)
            F = root_states.shape[-1]
            total = root_states.shape[0]
            if total % self.num_envs == 0:
                inferred_agents = total // self.num_envs
                root_states = root_states.view(self.num_envs, inferred_agents, F)
                self.agent_count = inferred_agents
            else:
                # fallback: assume agents-first
                root_states = root_states.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # The usual Isaac root layout: [px,py,pz, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
        F = root_states.shape[-1]
        if F >= 13:
            pos   = root_states[..., 0:3]
            quat  = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:13]
        elif F == 12:
            # Sometimes quat last or missing one comp; make a best-effort split
            pos   = root_states[..., 0:3]
            quat  = root_states[..., 3:7]
            lin_v = root_states[..., 7:10]
            ang_v = root_states[..., 10:12]
            # pad ang_v to 3
            pad = torch.zeros((*ang_v.shape[:-1], 1), device=self.device)
            ang_v = torch.cat([ang_v, pad], dim=-1)
        else:
            # Minimal fallback: positions only; zero the rest
            pos   = root_states[..., 0:3]
            quat  = torch.zeros((*pos.shape[:-1], 4), device=self.device); quat[..., -1] = 1.0
            lin_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)
            ang_v = torch.zeros((*pos.shape[:-1], 3), device=self.device)

        pose = torch.cat([pos, quat], dim=-1)
        vel  = torch.cat([lin_v, ang_v], dim=-1)

        # IMU (optional)
        imu = None
        if hasattr(rm, "imu_buffer"):
            imu_buf = rm.imu_buffer
            imu = imu_buf.view(self.num_envs, self.agent_count, -1).to(self.device)
        elif hasattr(rm, "accel_buffer") and hasattr(rm, "gyro_buffer"):
            acc  = rm.accel_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            gyro = rm.gyro_buffer.view(self.num_envs, self.agent_count, -1).to(self.device)
            imu  = torch.cat([acc, gyro], dim=-1)
        elif hasattr(rm, "get_imu"):
            imu = rm.get_imu()
            if not isinstance(imu, torch.Tensor):
                imu = torch.as_tensor(imu, device=self.device)
            if imu.dim() == 2:
                imu = imu.view(self.num_envs, self.agent_count, -1)

        if imu is None:
            imu = torch.zeros((self.num_envs, self.agent_count, 6), device=self.device)

        # One-time info
        if not hasattr(self, "_logged_state_source"):
            self._logged_state_source = True
            print(f"[CustomTask] Using root states from: {src_name}, shape={tuple(root_states.shape)}")

        return pose, vel, imu
    
    def _get_commanded_vz_envwise(self):
        rm = getattr(self.sim_env, "robot_manager", None)
        if rm is None or not hasattr(rm, "actions"):
            return torch.zeros((self.num_envs, 1), device=self.device)
        act = rm.actions
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act, device=self.device)
        act = act.to(self.device).float()
        # expect (agents, 4) -> take column 2 (vz), average across agents → (1,) then expand to (N,1)
        if act.dim() == 2 and act.shape[1] >= 3:
            vz_env = act[:, 2].mean().view(1, 1).repeat(self.num_envs, 1)
            return vz_env
        return torch.zeros((self.num_envs, 1), device=self.device)


    def _get_collisions_envwise(self) -> torch.Tensor:
        c = None
        if hasattr(self.sim_env, "collision_tensor"):
            c = self.sim_env.collision_tensor
        elif hasattr(self.sim_env, "robot_manager") and hasattr(self.sim_env.robot_manager, "collision_tensor"):
            c = self.sim_env.robot_manager.collision_tensor

        if c is None:
            # fallback: no collision info
            return torch.zeros((self.num_envs, 1), device=self.device)

        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c, device=self.device)

        c = c.to(self.device).float()

        # Normalize shape to (num_envs, 1):
        if c.dim() == 0:
            c = c.view(1, 1).repeat(self.num_envs, 1)
        elif c.dim() == 1:
            # Could be (num_envs,) or (agent_count,)
            if c.shape[0] == self.num_envs:
                c = c.view(self.num_envs, 1)
            else:
                # interpret as per-agent, reduce to per-env by any collision
                per_env_any = (c.view(1, -1) > 0).any(dim=1, keepdim=True).float()
                c = per_env_any.repeat(self.num_envs, 1)  # single env replicated
        elif c.dim() == 2:
            # If shape is (num_envs, agent_count), reduce to any-agent collision per env
            if c.shape[0] == self.num_envs:
                c = (c > 0).any(dim=1, keepdim=True).float()
            else:
                # Unknown layout; safest fallback
                c = torch.zeros((self.num_envs, 1), device=self.device)

        else:
            c = torch.zeros((self.num_envs, 1), device=self.device)

        return c
    

    def _get_terminated_truncated(self):
        N = self.num_envs
        # terminated: env-level collisions if available
        collisions_env = self._get_collisions_envwise()     # (N,1)
        terminated = (collisions_env > 0).view(N)           # (N,), bool

        # truncated: time limit
        limit = getattr(self.task_config, "episode_len_steps", 800)
        truncated = (self._steps >= limit)                   # (N,), bool
        return terminated, truncated



    def _update_coverage(self, pose):
        positions = pose[:, :, :3]  # shape: (envs, agents, 3)
        grid_size = 0.5

        indices = torch.clamp((positions[..., :2] / grid_size + 50).long(), 0, 99)  # (envs, agents, 2)
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.agent_count):
                x, y = indices[env_idx, agent_idx]
                self.coverage_maps[env_idx, agent_idx, x, y] = 1.0

    def _compute_rewards(self):
        N, A = self.num_envs, self.agent_count

        # 1) collisions -> (N,1)
        c = None
        if hasattr(self.sim_env, "collision_tensor"):
            c = self.sim_env.collision_tensor
        elif hasattr(self.sim_env, "robot_manager") and hasattr(self.sim_env.robot_manager, "collision_tensor"):
            c = self.sim_env.robot_manager.collision_tensor

        if c is None:
            collisions_env = torch.zeros((N, 1), device=self.device)
        else:
            if not isinstance(c, torch.Tensor):
                c = torch.as_tensor(c, device=self.device)
            c = c.to(self.device).float()
            # normalize to (N,1)
            if c.dim() == 0:
                collisions_env = c.view(1, 1).repeat(N, 1)
            elif c.dim() == 1:
                if c.shape[0] == N:
                    collisions_env = c.view(N, 1)
                elif c.shape[0] == A:
                    # per-agent -> any collision per env
                    collisions_env = (c.view(1, A) > 0).any(dim=1, keepdim=True).float().repeat(N, 1)
                else:
                    collisions_env = torch.zeros((N, 1), device=self.device)
            elif c.dim() == 2 and c.shape == (N, A):
                collisions_env = (c > 0).any(dim=1, keepdim=True).float()
            else:
                collisions_env = torch.zeros((N, 1), device=self.device)

        # 2) exploration -> (N,1)
        # coverage_maps: (N, A, H, W) -> any agent explored (N,H,W)
        explored_any = (self.coverage_maps > 0).any(dim=1).float()              # (N,H,W)
        explored_frac = explored_any.mean(dim=(1, 2)).unsqueeze(1)              # (N,1)

        # 3) update altitude proxy
        vz_env = self._get_commanded_vz_envwise()           # (N,1) in policy units
        # if you scaled actions to m/s already, great; else assume [-1,1] maps to ~1 m/s:
        vz_mps = vz_env * 1.0
        self._est_z = torch.clamp(self._est_z + vz_mps * self._dt, 0.0, self._z_max)

        # 4) altitude shaping around 1.5 m
        r_alt = -torch.abs(self._est_z - self._z_ref)       # (N,1)

        # 5) anti-fall from commands (helps early learning)
        up_bonus = torch.relu(vz_env)
        down_pen = torch.relu(-vz_env)
        anti_fall = 0.5 * up_bonus - 1.0 * down_pen
        # 6) compose env-level reward (N,1)
        reward = 10.0 * explored_frac - 5.0 * collisions_env + 0.5 * r_alt + anti_fall
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)                   # (N,1)

        # 7) write back
        self.task_obs["collisions"].copy_(collisions_env)  # (N,1)
        self.task_obs["rewards"].copy_(reward)             # (N,1)





@torch.jit.script
def compute_reward(
    pos_error: torch.Tensor,
    collisions: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    curriculum_level_multiplier: float,
    parameter_dict: TypingDict[str, float]
) -> torch.Tensor:
    reward = -pos_error - 5.0 * collisions
    return reward
