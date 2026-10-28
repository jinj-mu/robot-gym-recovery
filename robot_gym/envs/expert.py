"""Scripted expert for toy pickup on the play table."""

from __future__ import annotations

import numpy as np

from robot_gym.envs.pick_toys import PickToysFloorEnv


def _dir_action(delta: np.ndarray, scale: float = 0.03) -> np.ndarray:
    step = np.zeros(4, dtype=np.float64)
    n = np.linalg.norm(delta)
    if n > 1e-6:
        step[:3] = np.clip(delta / scale, -1.0, 1.0)
    return step


class ToyPickExpert:
    def __init__(self, env: PickToysFloorEnv):
        self.env = env
        self.phase = "hover"
        self._hold = 0

    def reset(self) -> None:
        self.phase = "hover"
        self._hold = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs
        ee = self.env._ee_pos()
        toy = self.env._toy_pos()
        bin_p = self.env._bin_pos()

        # Fingers sit ~5–6 cm below the hand body with the home wrist pose.
        hover = np.array([toy[0], toy[1], toy[2] + 0.20])
        pre = np.array([toy[0], toy[1], toy[2] + 0.12])
        grasp = np.array([toy[0], toy[1], toy[2] + 0.05])
        lift = np.array([toy[0], toy[1], toy[2] + 0.22])
        above_bin = np.array([bin_p[0], bin_p[1], bin_p[2] + 0.20])
        drop = np.array([bin_p[0], bin_p[1], bin_p[2] + 0.08])

        if self.phase == "hover":
            action = _dir_action(hover - ee)
            action[3] = -1.0
            if np.linalg.norm(ee - hover) < 0.04:
                self.phase = "pregrasp"
        elif self.phase == "pregrasp":
            action = _dir_action(pre - ee)
            action[3] = -1.0
            if np.linalg.norm(ee - pre) < 0.035:
                self.phase = "grasp"
        elif self.phase == "grasp":
            action = _dir_action(grasp - ee)
            action[3] = -1.0
            if np.linalg.norm(ee - grasp) < 0.04:
                self.phase = "close"
                self._hold = 20
        elif self.phase == "close":
            action = np.zeros(4)
            action[3] = 1.0
            self._hold -= 1
            if self._hold <= 0:
                self.phase = "lift"
        elif self.phase == "lift":
            action = _dir_action(lift - ee)
            action[3] = 1.0
            if self.env._toy_pos()[2] > 0.32 or ee[2] > lift[2] - 0.04:
                self.phase = "to_bin"
        elif self.phase == "to_bin":
            action = _dir_action(above_bin - ee)
            action[3] = 1.0
            if np.linalg.norm(ee[:2] - above_bin[:2]) < 0.05:
                self.phase = "lower"
        elif self.phase == "lower":
            action = _dir_action(drop - ee)
            action[3] = 1.0
            if ee[2] < drop[2] + 0.04:
                self.phase = "release"
                self._hold = 15
        elif self.phase == "release":
            action = np.zeros(4)
            action[3] = -1.0
            self._hold -= 1
            if self._hold <= 0:
                self.phase = "retreat"
        elif self.phase == "retreat":
            action = _dir_action(above_bin - ee)
            action[3] = -1.0
            if ee[2] > above_bin[2] - 0.03:
                self.phase = "done"
        else:
            action = np.zeros(4)
            action[3] = -1.0

        return action.astype(np.float32)
