"""
PyBullet + Stable-Baselines3 — Franka Reach RL Training
=========================================================
GPU  : RTX 4050 6 GB
VRAM : ~1–3 GB for neural net (PyTorch)  +  ~0 for PyBullet
       Total: comfortably under 6 GB

What this does:
  1. Trains a PPO policy to reach a target with the Kuka arm
  2. Shows a live matplotlib dashboard: reward curve, loss curve, VRAM bar
  3. After training, renders the learned policy in the GUI

Usage:
  conda activate robot_sim_env
  python robot_sim_demo/rl_reach.py
  python robot_sim_demo/rl_reach.py --render    # show GUI while training (slower)
  python robot_sim_demo/rl_reach.py --eval      # skip training, show policy
"""

import argparse
import math
import time
import sys
import os
import threading
import numpy as np

# ── Guards ────────────────────────────────────────────────────────────────────
for pkg in ["pybullet", "gymnasium", "stable_baselines3", "torch"]:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"ERROR: '{pkg}' not found. Run: conda activate robot_sim_env")
        sys.exit(1)

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import torch

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("[INFO] matplotlib unavailable — dashboard disabled")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────────────────────────────────────
# Custom Gym environment: Kuka arm reach task
# ─────────────────────────────────────────────────────────────────────────────
class KukaReachEnv(gym.Env):
    """
    Kuka IIWA arm must move its end-effector to a randomly placed target.

    Observation (9,):  joint angles (7) + EE position (x, y, z offset from target) → 10
    Action     (7,):  normalised joint angle deltas
    Reward          :  -distance  +  bonus on success  -  time penalty
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self._client = -1
        self._robot_id = -1
        self._target_id = -1
        self._step_count = 0
        self._max_steps = 150
        self._n_joints = 7

        # Action: delta joint angles [-1, 1] normalised
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._n_joints,),
            dtype=np.float32
        )

        # Obs: joint angles + EE xyz displacement to target
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._n_joints + 3,),
            dtype=np.float32
        )

        # Joint limits (Kuka IIWA approximate)
        self._jlimits = [
            (-2.97, 2.97), (-2.09, 2.09), (-2.97, 2.97),
            (-2.09, 2.09), (-2.97, 2.97), (-2.09, 2.09),
            (-3.05, 3.05)
        ]

    def _make_client(self):
        if self._client >= 0:
            return
        if self.render_mode == "human":
            cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.5, 60, -30, [0, 0, 0.5],
                                         physicsClientId=cid)
        else:
            cid = p.connect(p.DIRECT)
        self._client = cid

    def _load_scene(self):
        cid = self._client
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane.urdf", physicsClientId=cid)
        self._robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=cid
        )

        # Target sphere (invisible in DIRECT, visible in GUI)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05,
                                  rgbaColor=[1, 0, 0, 0.8],
                                  physicsClientId=cid)
        self._target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[0.5, 0, 0.5],
            physicsClientId=cid
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._client < 0:
            self._make_client()
            self._load_scene()
        else:
            # Reset joint positions
            for j in range(self._n_joints):
                angle = np.random.uniform(*self._jlimits[j]) * 0.1
                p.resetJointState(self._robot_id, j, angle,
                                  physicsClientId=self._client)

        # Random target in workspace
        tx = np.random.uniform(0.3, 0.6)
        ty = np.random.uniform(-0.3, 0.3)
        tz = np.random.uniform(0.3, 0.7)
        self._target = np.array([tx, ty, tz])
        p.resetBasePositionAndOrientation(
            self._target_id, [tx, ty, tz],
            [0, 0, 0, 1], physicsClientId=self._client
        )

        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        cid = self._client
        action = np.clip(action, -1, 1)

        # Apply delta joint angles
        for j in range(self._n_joints):
            cur = p.getJointState(self._robot_id, j, physicsClientId=cid)[0]
            lo, hi = self._jlimits[j]
            delta = action[j] * 0.05   # max 0.05 rad per step
            target = np.clip(cur + delta, lo, hi)
            p.setJointMotorControl2(
                self._robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200,
                physicsClientId=cid
            )

        for _ in range(4):   # 4 sub-steps per action step
            p.stepSimulation(physicsClientId=cid)

        obs = self._obs()
        ee_pos = self._ee_pos()
        dist = float(np.linalg.norm(ee_pos - self._target))

        # Reward
        reward = -dist                      # distance penalty
        if dist < 0.05:
            reward += 10.0                  # success bonus
        reward -= 0.005                     # time penalty

        self._step_count += 1
        terminated = bool(dist < 0.05)
        truncated  = bool(self._step_count >= self._max_steps)

        return obs, reward, terminated, truncated, {"distance": dist}

    def _ee_pos(self):
        ee = p.getLinkState(self._robot_id, self._n_joints - 1,
                            physicsClientId=self._client)
        return np.array(ee[0], dtype=np.float32)

    def _obs(self):
        joints = np.array([
            p.getJointState(self._robot_id, j, physicsClientId=self._client)[0]
            for j in range(self._n_joints)
        ], dtype=np.float32)
        ee = self._ee_pos()
        disp = ee - self._target.astype(np.float32)
        return np.concatenate([joints, disp]).astype(np.float32)

    def render(self):
        pass   # PyBullet GUI mode handles its own rendering

    def close(self):
        if self._client >= 0:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = -1


# ─────────────────────────────────────────────────────────────────────────────
# Live matplotlib dashboard
# ─────────────────────────────────────────────────────────────────────────────
class LiveDashboard:
    """Matplotlib window showing reward, loss, and VRAM usage."""

    def __init__(self):
        if not HAS_MPL:
            return
        self.rewards     = []
        self.ep_lens     = []
        self.distances   = []
        self.vram_used   = []
        self.timestamps  = []
        self._lock       = threading.Lock()

        plt.ion()
        self.fig = plt.figure(figsize=(11, 7), facecolor="#0f0f23")
        self.fig.suptitle(
            "Robot RL Training Dashboard — RTX 4050 6 GB",
            color="white", fontsize=13, fontweight="bold"
        )
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               hspace=0.45, wspace=0.35)

        ax_kw = dict(facecolor="#1a1a2e")
        self.ax_rew  = self.fig.add_subplot(gs[0, :2], **ax_kw)
        self.ax_dist = self.fig.add_subplot(gs[1, :2], **ax_kw)
        self.ax_vram = self.fig.add_subplot(gs[0, 2],  **ax_kw)
        self.ax_stat = self.fig.add_subplot(gs[1, 2],  **ax_kw)

        for ax, title in [
            (self.ax_rew,  "Episode Reward"),
            (self.ax_dist, "Final Distance to Target (m)"),
            (self.ax_vram, "GPU VRAM Usage"),
            (self.ax_stat, "Stats"),
        ]:
            ax.set_title(title, color="#aad4f5", fontsize=10)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")

        self.ax_stat.axis("off")
        plt.show()

    def update(self, episode, reward, dist, ep_len):
        if not HAS_MPL:
            return
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated(0) / 1e9

        with self._lock:
            self.rewards.append(reward)
            self.distances.append(dist)
            self.ep_lens.append(ep_len)
            self.vram_used.append(vram)
            self.timestamps.append(episode)

        self._redraw(episode, vram)

    def _redraw(self, episode, vram):
        try:
            xs = self.timestamps

            self.ax_rew.clear()
            self.ax_rew.set_facecolor("#1a1a2e")
            self.ax_rew.set_title("Episode Reward", color="#aad4f5", fontsize=10)
            if len(xs) > 1:
                self.ax_rew.plot(xs, self.rewards, color="#4fc3f7", lw=1.5, alpha=0.7)
                # Rolling average
                w = min(20, len(self.rewards))
                avg = np.convolve(self.rewards, np.ones(w)/w, mode="valid")
                self.ax_rew.plot(xs[w-1:], avg, color="#ff9800", lw=2.5, label="avg")
                self.ax_rew.legend(fontsize=8, labelcolor="white",
                                   facecolor="#1a1a2e")
            self.ax_rew.axhline(0, color="#555", lw=0.5)
            self.ax_rew.tick_params(colors="white")

            self.ax_dist.clear()
            self.ax_dist.set_facecolor("#1a1a2e")
            self.ax_dist.set_title("Final Distance to Target (m)",
                                   color="#aad4f5", fontsize=10)
            if len(xs) > 1:
                self.ax_dist.plot(xs, self.distances, color="#a5d6a7",
                                  lw=1.5, alpha=0.7)
                self.ax_dist.axhline(0.05, color="#ff5252", lw=1,
                                     linestyle="--", label="success threshold")
                self.ax_dist.legend(fontsize=8, labelcolor="white",
                                    facecolor="#1a1a2e")
            self.ax_dist.tick_params(colors="white")

            # VRAM bar
            self.ax_vram.clear()
            self.ax_vram.set_facecolor("#1a1a2e")
            self.ax_vram.set_title("GPU VRAM (GB)", color="#aad4f5", fontsize=10)
            total = 6.0  # RTX 4050 6 GB
            color = "#ef5350" if vram / total > 0.85 else "#66bb6a"
            self.ax_vram.barh(["Used"], [vram], color=color)
            self.ax_vram.barh(["Total"], [total], color="#263238")
            self.ax_vram.set_xlim(0, total)
            self.ax_vram.text(vram + 0.1, 0, f"{vram:.2f} GB",
                              va="center", color="white", fontsize=9)
            self.ax_vram.tick_params(colors="white")

            # Text stats
            self.ax_stat.clear()
            self.ax_stat.set_facecolor("#1a1a2e")
            self.ax_stat.axis("off")
            lines = [
                f"Episode  : {episode}",
                f"Reward   : {self.rewards[-1]:.3f}" if self.rewards else "",
                f"Avg(20)  : {np.mean(self.rewards[-20:]):.3f}" if len(self.rewards) >= 2 else "",
                f"Dist     : {self.distances[-1]:.4f} m" if self.distances else "",
                f"Success  : {'✓' if (self.distances and self.distances[-1] < 0.05) else '✗'}",
                f"VRAM     : {vram:.2f} / {total:.1f} GB",
                f"GPU      : RTX 4050 Laptop",
            ]
            for i, line in enumerate(lines):
                self.ax_stat.text(
                    0.05, 0.92 - i * 0.13, line,
                    transform=self.ax_stat.transAxes,
                    color="white", fontsize=9.5,
                    fontfamily="monospace"
                )

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# SB3 callback — updates dashboard after each episode
# ─────────────────────────────────────────────────────────────────────────────
class DashboardCallback(BaseCallback):
    def __init__(self, dashboard: LiveDashboard, verbose=0):
        super().__init__(verbose)
        self.dashboard   = dashboard
        self._episode    = 0
        self._ep_rewards = []
        self._ep_dists   = []

    def _on_step(self) -> bool:
        # Read info from Monitor wrapper
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode += 1
                reward = info["episode"]["r"]
                dist   = info.get("distance", 0.0)
                ep_len = info["episode"]["l"]
                self.dashboard.update(self._episode, reward, dist, ep_len)

                if self._episode % 50 == 0:
                    print(f"  Ep {self._episode:4d} | "
                          f"reward={reward:7.3f} | dist={dist:.4f}m")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — runs trained policy with GUI rendering
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(model, n_episodes=5):
    print("\n" + "═" * 54)
    print("  Evaluating learned policy in GUI...")
    print("  Close the PyBullet window to exit.")
    print("═" * 54)

    env_gui = KukaReachEnv(render_mode="human")
    obs, _ = env_gui.reset()

    for ep in range(n_episodes):
        obs, _ = env_gui.reset()
        total_r = 0
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env_gui.step(action)
            total_r += r
            time.sleep(1.0 / 60.0)
            if term or trunc:
                break
        dist = info.get("distance", "?")
        print(f"  Episode {ep+1}: reward={total_r:.2f}  final_dist={dist}")

    env_gui.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="Show PyBullet GUI during training (slower)")
    parser.add_argument("--eval",   action="store_true",
                        help="Skip training, just run eval with saved model")
    parser.add_argument("--steps",  type=int, default=200_000,
                        help="Total training steps (default 200000)")
    parser.add_argument("--model",  type=str, default="kuka_ppo",
                        help="Model save path prefix")
    args = parser.parse_args()

    print("═" * 54)
    print("  Kuka Reach RL Training")
    print(f"  GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    print(f"  Steps: {args.steps:,}")
    print("═" * 54 + "\n")

    model_path = f"{args.model}.zip"

    if args.eval:
        if not os.path.exists(model_path):
            print(f"No saved model at {model_path}. Train first.")
            sys.exit(1)
        model = PPO.load(model_path)
        evaluate_policy(model)
        return

    # ── Training ──────────────────────────────────────────────────────────────
    render_mode = "human" if args.render else None
    env = Monitor(KukaReachEnv(render_mode=render_mode))

    print("[*] Checking environment...")
    try:
        check_env(env, warn=True)
        print("  ✓ Env passed gym check")
    except Exception as e:
        print(f"  Warning: {e}")

    print("[*] Building PPO model (MLP policy)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tb_logs/",
    )

    # Check VRAM after model creation
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"  VRAM after model init: {vram_used:.3f} GB")

    dashboard = LiveDashboard()
    callback  = DashboardCallback(dashboard)

    print(f"\n[*] Training for {args.steps:,} steps...")
    print("    Live dashboard: matplotlib window (if available)")
    print("    Press Ctrl+C to stop early and save.\n")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted — saving model...")

    model.save(model_path)
    print(f"\n  ✓ Model saved to {model_path}")

    # ── Post-training evaluation ──────────────────────────────────────────────
    env.close()
    evaluate_policy(model)

    if HAS_MPL:
        plt.ioff()
        input("\n  Press Enter to close the dashboard...")
        plt.close("all")


if __name__ == "__main__":
    main()
