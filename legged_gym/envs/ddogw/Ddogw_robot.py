import torch

from legged_gym.envs import LeggedRobot
from .Ddogw_config import DdogwRoughCfg


class Ddogw(LeggedRobot):
    cfg: DdogwRoughCfg

    def _init_buffers(self):
        super()._init_buffers()
        self.stumble_memory_steps = self.cfg.rewards.stumble_memory_steps
        self.stumble_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )

    def _get_feet_stumble_mask(self):
        """与 _reward_feet_stumble 相同：轮子接触力横向远大于竖向。"""
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 3.0 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _update_stumble_timer(self):
        stumble_now = self._get_feet_stumble_mask()
        self.stumble_timer = torch.where(
            stumble_now,
            torch.full_like(self.stumble_timer, self.stumble_memory_steps),
            torch.clamp(self.stumble_timer - 1, min=0),
        )

    def compute_reward(self):
        self._update_stumble_timer()
        super().compute_reward()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.stumble_timer[env_ids] = 0

    def _reward_stair_bounce(self):
        """撞台阶后的短时间内，若指令向前但机身系前向速度为负，按后退速度惩罚。"""
        stumble_recent = self.stumble_timer > 0
        cmd_forward = self.commands[:, 0] > self.cfg.rewards.cmd_forward_threshold
        backward = torch.clamp(-self.base_lin_vel[:, 0], min=0.0)
        return stumble_recent.float() * cmd_forward.float() * backward
