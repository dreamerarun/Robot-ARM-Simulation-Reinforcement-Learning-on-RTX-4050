# 🤖 Robot Simulation & Reinforcement Learning on RTX 4050

A lightweight robotics simulation and reinforcement-learning project built with **PyBullet, PyTorch, Gymnasium, and Stable-Baselines3**.

The project demonstrates two complementary robotics workflows:

* 🦾 **Franka Panda simulation** with inverse kinematics, joint control, target tracking, camera visualization, and contact-force visualization.
* 🧠 **Kuka IIWA reinforcement learning** using PPO to learn an end-effector reaching task.
* 📊 **Live training dashboard** for monitoring reward, target distance, and GPU VRAM usage.
* 🎮 **Interactive PyBullet GUI** for visualizing robot behavior and manually controlling joints.

The project is specifically configured for laptops with limited GPU memory, such as the **NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM**.

---

## ✨ Features

### 🦾 Franka Panda Simulation

The Franka simulation provides a real-time 3D PyBullet environment containing:

* Franka Panda robotic arm
* Ground plane
* Table
* Moving target sphere
* Small cube object
* Inverse kinematics control
* Individual joint debug sliders
* Real-time physics simulation
* Contact-force visualization
* Camera feed through Matplotlib

The robot continuously calculates inverse kinematics to move its end-effector toward a moving target.

### 🧠 Kuka IIWA Reinforcement Learning

The reinforcement-learning environment uses **Stable-Baselines3 PPO** to train a Kuka IIWA robot to reach randomly positioned targets.

The custom Gymnasium environment contains:

* 7-dimensional joint-angle state
* 3-dimensional end-effector displacement
* 7-dimensional normalized joint-action space
* Randomized target positions
* Joint-limit constraints
* Distance-based reward
* Success bonus
* Time penalty
* Episode termination when the target is reached

The target is considered reached when the end-effector comes within **5 cm** of the target.

---

# 🏗️ Project Architecture

```text
Robot Simulation Project
│
├── PyBullet Simulation
│   │
│   ├── Franka Panda
│   ├── Inverse Kinematics
│   ├── Joint Control
│   ├── Moving Target
│   ├── Contact Detection
│   └── Camera Visualization
│
└── Reinforcement Learning
    │
    ├── Kuka IIWA
    ├── Gymnasium Environment
    ├── PPO Algorithm
    ├── PyTorch
    ├── Target Reaching
    └── Live Training Dashboard
```

---

# 📁 Project Structure

```text
robot-simulation/
│
├── franka_sim.py
│   └── Franka Panda PyBullet simulation
│
├── rl_reach.py
│   └── Kuka IIWA PPO reinforcement-learning environment
│
├── setup_robot_sim.sh
│   └── Automated environment setup script
│
├── README.md
│   └── Project documentation
│
├── kuka_ppo.zip
│   └── Saved PPO model generated after training
│
└── tb_logs/
    └── TensorBoard training logs
```

> `kuka_ppo.zip` and `tb_logs/` are generated during training and do not need to be committed to GitHub unless you want to publish the trained model/logs.

---

# 💻 Hardware

The project was designed around an:

| Component       | Configuration                      |
| --------------- | ---------------------------------- |
| GPU             | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM            | 6 GB                               |
| Physics Engine  | PyBullet                           |
| Deep Learning   | PyTorch                            |
| RL Framework    | Stable-Baselines3                  |
| Environment API | Gymnasium                          |
| Visualization   | PyBullet GUI + Matplotlib          |

The provided setup script checks for both **Conda** and an NVIDIA GPU using `nvidia-smi` before creating the environment.

---

# 🧠 Why PyBullet?

This project uses PyBullet rather than a heavier simulator because the target hardware has only **6 GB of VRAM**.

The project README estimates that:

* PyBullet physics runs primarily on the CPU.
* The PPO neural network requires roughly 1–2 GB of VRAM.
* Isaac Sim's GUI/rendering requirements can exceed the available VRAM on a 6 GB laptop GPU.

Therefore, PyBullet provides a practical environment for experimenting with robotics and reinforcement learning on hardware with limited GPU memory.

---

# ⚙️ Installation

## 1. Prerequisites

Install:

* NVIDIA GPU driver
* Miniconda or Anaconda
* Git

Verify Conda:

```bash
conda --version
```

Verify the NVIDIA GPU:

```bash
nvidia-smi
```

The setup script specifically checks that both commands are available before proceeding.

---

## 2. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

---

## 3. Run the Automated Setup

Make the setup script executable:

```bash
chmod +x setup_robot_sim.sh
```

Run it:

```bash
./setup_robot_sim.sh
```

The script creates an isolated Conda environment called:

```text
robot_sim_env
```

with Python 3.11.

It installs the main project dependencies, including PyBullet, PyTorch, Stable-Baselines3, Gymnasium, NumPy, Matplotlib, OpenCV, Open3D, and Rich.

---

# 📦 Main Dependencies

The setup script installs:

```text
Python 3.11
PyBullet
PyTorch 2.2.2
TorchVision 0.17.2
Stable-Baselines3 2.3.0
Gymnasium 0.29.1
NumPy
Matplotlib
OpenCV
Open3D
Rich
```

PyTorch is installed using the CUDA 12.1 wheel specified by the setup script.

---

# 🦾 Demo 1 — Franka Panda Simulation

Activate the environment:

```bash
conda activate robot_sim_env
```

Run:

```bash
python franka_sim.py
```

The simulation launches a PyBullet GUI containing a Franka Panda arm, table, moving target, and cube.

## What happens?

The target moves continuously in 3D:

```text
        Target
          ●
         ↗
        /
   ┌─────────┐
   │  Franka │
   │   Arm   │
   └─────────┘
```

The robot calculates inverse kinematics:

```text
Moving Target
      │
      ▼
Target Position
      │
      ▼
Inverse Kinematics
      │
      ▼
Joint Positions
      │
      ▼
Position Controllers
      │
      ▼
Franka End-Effector
```

The IK solver uses PyBullet's `calculateInverseKinematics()` to determine joint configurations for the target position.

---

# 🎮 Franka Controls

Inside the PyBullet window:

| Control       | Action                  |
| ------------- | ----------------------- |
| Mouse drag    | Rotate camera           |
| Scroll        | Zoom                    |
| Joint sliders | Manually control joints |
| `q`           | Quit                    |

The project creates debug sliders for the robot's revolute joints.

---

# 📷 Camera Visualization

The Franka simulation also creates a separate Matplotlib camera window.

The camera:

* Orbits around the robot
* Captures 640 × 360 frames
* Uses PyBullet's CPU-based Tiny Renderer
* Updates at approximately 20 FPS
* Displays the camera feed through Matplotlib

This keeps the camera rendering lightweight for the RTX 4050 configuration.

---

# 🧠 Demo 2 — Kuka PPO Reinforcement Learning

Run the standard training:

```bash
python rl_reach.py
```

Default training:

```text
200,000 environment steps
```

The script exposes the number of training steps through the `--steps` argument.

---

## Quick Training Test

For a shorter experiment:

```bash
python rl_reach.py --steps 50000
```

---

## Training With PyBullet GUI

To visualize the simulation during training:

```bash
python rl_reach.py --render --steps 50000
```

Rendering makes training slower because the simulation must maintain the GUI.

---

## Evaluate a Trained Model

After training:

```bash
python rl_reach.py --eval
```

The saved model is loaded and the learned policy is executed in the PyBullet GUI.

---

# 🧩 Reinforcement Learning Environment

The custom environment is:

```python
KukaReachEnv
```

It follows the Gymnasium environment interface.

## Observation Space

The observation contains:

```text
7 × joint angles
+
3 × end-effector displacement
```

Therefore:

```text
Observation = 10 dimensions
```

The displacement is calculated as:

```text
EE Position - Target Position
```

## The implementation constructs this observation directly from the Kuka joint states and end-effector position.

# 🎯 Action Space

The PPO policy outputs:

```text
7 normalized joint actions
```

Each action is constrained to:

```text
[-1, +1]
```

The normalized action is converted into a joint-angle change:

```text
Δθ = action × 0.05 rad
```

## and then constrained by the approximate Kuka joint limits.

# 🏆 Reward Function

The reward is based on the distance between the end-effector and target.

```text
Reward = -distance
         + success bonus
         - time penalty
```

Specifically:

```python
reward = -dist

if dist < 0.05:
    reward += 10.0

reward -= 0.005
```

The 5 cm threshold also terminates the episode when the target is successfully reached.

This encourages the robot to:

1. Minimize end-effector distance.
2. Reach the target within 5 cm.
3. Complete the task efficiently.

---

# 🤖 PPO Configuration

The project uses Stable-Baselines3 PPO with an MLP policy.

Main parameters:

```python
PPO(
    "MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,
)
```

The implementation automatically selects CUDA when PyTorch detects a CUDA-capable GPU; otherwise it falls back to CPU.

---

# 📊 Live Training Dashboard

During training, the project provides a Matplotlib dashboard containing:

### 1. Episode Reward

Tracks the reward obtained by the policy over episodes.

### 2. Final Distance

Shows the final end-effector distance from the target.

A **5 cm line** indicates the success threshold.

### 3. GPU VRAM

Displays the amount of CUDA memory allocated by PyTorch.

### 4. Statistics

Displays:

```text
Episode
Reward
20-episode average
Distance
Success status
VRAM usage
GPU
```

## The dashboard is updated through a Stable-Baselines3 callback after completed episodes.

# 💾 Model Saving

After training, the PPO model is saved as:

```text
kuka_ppo.zip
```

You can specify a different model name:

```bash
python rl_reach.py --model my_robot_model
```

This produces:

```text
my_robot_model.zip
```

The default save path is defined by the `--model` argument.

---

# 📈 TensorBoard

Training logs are configured to be written to:

```text
tb_logs/
```

You can launch TensorBoard with:

```bash
tensorboard --logdir tb_logs/
```

Then open the displayed local TensorBoard address in your browser.

---

# 🔬 Simulation vs Reinforcement Learning

The project demonstrates two different approaches to robot control.

| Feature               | Franka Simulation  | Kuka RL    |
| --------------------- | ------------------ | ---------- |
| Robot                 | Franka Panda       | Kuka IIWA  |
| Control               | Inverse Kinematics | PPO        |
| Target                | Moving             | Randomized |
| Learning              | ❌                  | ✅          |
| Manual control        | ✅                  | ❌          |
| PyBullet GUI          | ✅                  | Optional   |
| Camera feed           | ✅                  | ❌          |
| Training dashboard    | ❌                  | ✅          |
| Contact visualization | ✅                  | ❌          |

---

# 🧪 Headless Training

For faster training, use:

```bash
python rl_reach.py
```

The default RL environment uses:

```python
p.connect(p.DIRECT)
```

when GUI rendering is disabled.

This avoids the PyBullet GUI while allowing PPO to interact with the physics simulation.

---

# ⚠️ GPU Memory Considerations

For an RTX 4050 Laptop GPU with 6 GB VRAM, the project is designed to keep GPU usage relatively low.

The provided project documentation estimates approximately:

```text
PPO network       ~1–2 GB
PyTorch buffers   ~400 MB
PyBullet physics  ~0 GB GPU
```

The exact VRAM usage depends on the installed PyTorch/CUDA environment and system workload.

### Important

Avoid simultaneously running other large GPU workloads such as large generative-AI models while training PPO.

---

# 🛠️ Troubleshooting

## PyBullet GUI does not open

Check:

```bash
echo $DISPLAY
```

If you are running in a headless environment, use the non-GUI mode or configure an appropriate display.

---

## `No module named 'pybullet'`

Activate the project environment:

```bash
conda activate robot_sim_env
```

Then verify:

```bash
python -c "import pybullet; print(pybullet.__version__)"
```

---

## CUDA Out of Memory

Try reducing the PPO batch size in `rl_reach.py`:

```python
batch_size=32
```

instead of:

```python
batch_size=64
```

You can also force CPU execution by changing:

```python
device="cuda"
```

to:

```python
device="cpu"
```

---

## Matplotlib Dashboard Does Not Appear

The project attempts to use the TkAgg backend.

On Ubuntu, install Tk:

```bash
conda activate robot_sim_env
conda install -c conda-forge tk
```

Then rerun the training script.

---

# 🧹 Removing the Environment

If you want to completely remove the project's Conda environment:

```bash
conda env remove -n robot_sim_env
```

Remove generated training artifacts:

```bash
rm -rf tb_logs/
rm -f kuka_ppo.zip
```

---

# 🚀 Future Improvements

Possible extensions for this project include:

* [ ] Add obstacle avoidance
* [ ] Add collision-aware rewards
* [ ] Train with randomized robot initial configurations
* [ ] Add domain randomization
* [ ] Add vision-based observations
* [ ] Add camera observations to the RL policy
* [ ] Add grasping and manipulation tasks
* [ ] Add object-pick-and-place tasks
* [ ] Compare PPO with SAC/TD3
* [ ] Add experiment tracking
* [ ] Export trained policy for deployment
* [ ] Integrate ROS 2
* [ ] Transfer the learned policy to a real robot

---

# 📚 Technologies Used

* **Python**
* **PyBullet**
* **Gymnasium**
* **Stable-Baselines3**
* **PyTorch**
* **NumPy**
* **Matplotlib**
* **OpenCV**
* **Open3D**
* **Conda**
* **CUDA**

---

# 🎓 Learning Objectives

This project provides hands-on experience with:

* Robot simulation
* Forward and inverse kinematics
* Joint-space control
* End-effector control
* Physics simulation
* Gymnasium environments
* Reinforcement learning
* PPO
* Reward design
* Robot state/action spaces
* GPU-accelerated machine learning
* Real-time visualization
* Model evaluation
* Resource monitoring

---

# 📌 Project Summary

This project combines **classical robot control** and **reinforcement learning** in a lightweight simulation environment.

The Franka Panda demonstration focuses on deterministic robot control using inverse kinematics, while the Kuka IIWA environment demonstrates how a PPO agent can learn an end-effector reaching behavior through interaction with a physics simulator.

The overall setup is intentionally lightweight enough to experiment with robotics and RL on an **RTX 4050 6 GB laptop GPU**, without requiring a high-end workstation.

---

## 👨‍💻 Author

**Arun M.**

Robotics & Automation | Physical AI | Robot Learning | NVIDIA Technologies

GitHub: `github.com/dreamerarun`

---


⭐ If this project helped you learn about robot simulation or reinforcement learning, consider giving the repository a star.
