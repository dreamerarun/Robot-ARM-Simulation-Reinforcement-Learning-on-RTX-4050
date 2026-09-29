"""
PyBullet Robot Simulation — Franka Panda Arm
=============================================
GPU  : RTX 4050 6 GB  (PyBullet uses OpenGL, not CUDA → ~0 VRAM used)
Mode : Full 3D GUI window with real-time physics

What this demo does:
  1. Opens a 3D OpenGL window with a Franka Panda arm
  2. Live sinusoidal joint-space motion with debug sliders
  3. Inverse kinematics to reach a moving target sphere
  4. Real-time camera feed rendered into a matplotlib window
  5. Contact force visualisation (debug lines)

Controls in the PyBullet window:
  • Mouse drag  — rotate camera
  • Scroll      — zoom
  • Sliders     — control individual joints
  • 'q'         — quit
"""

import math
import time
import sys
import os
import numpy as np

# ── Guard ─────────────────────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not found.")
    print("Run:  conda activate robot_sim_env")
    sys.exit(1)

# ── Optional matplotlib for the camera overlay ────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")          # works on Ubuntu with display
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("[INFO] matplotlib not available — skipping camera overlay window")

import threading
import queue


# ─────────────────────────────────────────────────────────────────────────────
# Scene setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_scene():
    """Load ground, robot, table, and target object."""

    # Ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    # Table
    table_id = p.loadURDF(
        "table/table.urdf",
        basePosition=[0.5, 0.0, 0.0],
        globalScaling=1.0
    )

    # Franka Panda arm
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.62],   # on table surface
        useFixedBase=True
    )

    # Target sphere (red — arm will try to reach this)
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.04,
        rgbaColor=[1.0, 0.2, 0.2, 0.9]
    )
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
    target_id = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.6, 0.0, 0.9]
    )

    # Small blue cube (grasp object)
    cube_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.025, 0.025, 0.025],
        rgbaColor=[0.2, 0.4, 0.9, 1.0]
    )
    cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_col,
        baseVisualShapeIndex=cube_vis,
        basePosition=[0.5, 0.15, 0.73]
    )

    return robot_id, target_id, cube_id


def setup_debug_sliders(robot_id, num_joints):
    """Create per-joint sliders in the PyBullet GUI."""
    sliders = []
    joint_names = []

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        jtype = info[2]
        jname = info[1].decode()

        if jtype == p.JOINT_REVOLUTE:
            lo, hi = info[8], info[9]
            if lo >= hi:   # unconstrained — use safe range
                lo, hi = -math.pi, math.pi
            slider = p.addUserDebugParameter(jname, lo, hi, 0.0)
            sliders.append((j, slider))
            joint_names.append(jname)

    return sliders


# ─────────────────────────────────────────────────────────────────────────────
# Inverse kinematics helper
# ─────────────────────────────────────────────────────────────────────────────
def ik_reach(robot_id, ee_link, target_pos, target_orn=None):
    """Compute IK joint angles to reach target_pos."""
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])  # pointing down

    joint_poses = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        target_pos,
        target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    return joint_poses


def apply_joint_poses(robot_id, joint_poses, controllable_joints):
    """Apply IK result to the robot."""
    for i, j in enumerate(controllable_joints):
        if i < len(joint_poses):
            p.setJointMotorControl2(
                robot_id, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=240.0,
                maxVelocity=0.5
            )


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture thread (feeds matplotlib window)
# ─────────────────────────────────────────────────────────────────────────────
def camera_thread_fn(frame_queue: queue.Queue, stop_event: threading.Event):
    """Runs in background — captures PyBullet camera frames."""
    W, H = 640, 360
    fov, near, far = 60, 0.1, 10.0

    while not stop_event.is_set():
        # Orbit camera
        t = time.time()
        cx = 0.5 + 0.6 * math.cos(t * 0.15)
        cy = 0.6 * math.sin(t * 0.15)
        cz = 1.1

        vm = p.computeViewMatrix(
            cameraEyePosition=[cx, cy, cz],
            cameraTargetPosition=[0.5, 0.0, 0.85],
            cameraUpVector=[0, 0, 1]
        )
        pm = p.computeProjectionMatrixFOV(fov, W / H, near, far)

        _, _, rgba, _, _ = p.getCameraImage(
            W, H,
            viewMatrix=vm,
            projectionMatrix=pm,
            renderer=p.ER_TINY_RENDERER     # CPU renderer — no VRAM
        )
        frame = np.array(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]

        # Non-blocking put
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        time.sleep(0.05)   # ~20 fps


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Launch PyBullet with GUI ──────────────────────────────────────────────
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)      # manual stepping — more control
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 0.8]
    )

    # Nicer background
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    print("═" * 54)
    print("  PyBullet Robot Simulation — RTX 4050 6 GB")
    print("  VRAM used by PyBullet : ~0 MB (OpenGL, CPU renderer)")
    print("═" * 54)

    # ── Build scene ───────────────────────────────────────────────────────────
    robot_id, target_id, cube_id = setup_scene()

    # Find controllable joints
    num_joints = p.getNumJoints(robot_id)
    ctrl_joints = [
        j for j in range(num_joints)
        if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE
    ]
    ee_link = num_joints - 1   # end-effector link index

    print(f"  Robot joints  : {num_joints}")
    print(f"  Controllable  : {len(ctrl_joints)}")
    print(f"  End-effector  : link {ee_link}")
    print("")

    # ── Debug sliders ─────────────────────────────────────────────────────────
    sliders = setup_debug_sliders(robot_id, num_joints)

    # Mode label in GUI
    mode_id = p.addUserDebugText(
        "Mode: IK REACH",
        [0.5, 0.0, 1.35],
        textColorRGB=[0.2, 0.9, 0.2],
        textSize=1.5
    )

    # ── Optional matplotlib camera window ─────────────────────────────────────
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=2)
    cam_thread = None
    fig, ax, im_handle = None, None, None

    if HAS_MPL:
        cam_thread = threading.Thread(
            target=camera_thread_fn,
            args=(frame_queue, stop_event),
            daemon=True
        )
        cam_thread.start()

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_title("Robot Camera Feed (CPU renderer)", color="white", fontsize=10)
        ax.axis("off")
        im_handle = ax.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
        plt.tight_layout()
        plt.ion()
        plt.show()

    # ── Simulation loop ───────────────────────────────────────────────────────
    t = 0.0
    step = 0
    contact_lines = []

    print("  Simulation running. Close the PyBullet window to quit.")
    print("  Use the sliders on the left to control joints manually.")
    print("")

    try:
        while True:
            # ── Target orbits in 3D ──────────────────────────────────────────
            tx = 0.5 + 0.18 * math.cos(t * 0.4)
            ty = 0.18 * math.sin(t * 0.4)
            tz = 0.85 + 0.08 * math.sin(t * 0.7)
            p.resetBasePositionAndOrientation(
                target_id, [tx, ty, tz],
                p.getQuaternionFromEuler([0, 0, 0])
            )

            # ── IK reach toward target ───────────────────────────────────────
            ik_poses = ik_reach(robot_id, ee_link, [tx, ty, tz])
            apply_joint_poses(robot_id, ik_poses, ctrl_joints)

            # ── Override with slider values if user touched them ─────────────
            # (sliders default to 0, so check if they've moved)
            for joint_idx, slider in sliders:
                val = p.readUserDebugParameter(slider)
                if abs(val) > 0.01:
                    p.setJointMotorControl2(
                        robot_id, joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=val,
                        force=240.0
                    )

            # ── Step physics ─────────────────────────────────────────────────
            p.stepSimulation()
            t += 1.0 / 240.0
            step += 1

            # ── Contact force debug lines ─────────────────────────────────────
            if step % 10 == 0:
                for line in contact_lines:
                    p.removeUserDebugItem(line)
                contact_lines.clear()

                contacts = p.getContactPoints(robot_id, cube_id)
                for c in (contacts or []):
                    pos = c[5]          # contact position on cube
                    nrm = c[7]          # contact normal
                    force = c[9]
                    end = [pos[i] + nrm[i] * force * 0.002 for i in range(3)]
                    lid = p.addUserDebugLine(
                        pos, end,
                        lineColorRGB=[1, 0.5, 0],
                        lineWidth=2,
                        lifeTime=0.1
                    )
                    contact_lines.append(lid)

            # ── Update matplotlib camera overlay ─────────────────────────────
            if HAS_MPL and step % 6 == 0:
                try:
                    frame = frame_queue.get_nowait()
                    im_handle.set_data(frame)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                except queue.Empty:
                    pass

            # ── Console status every 240 steps (1 sec) ───────────────────────
            if step % 240 == 0:
                ee_state = p.getLinkState(robot_id, ee_link)
                ee_pos = ee_state[0]
                dist = math.sqrt(sum((ee_pos[i] - [tx, ty, tz][i]) ** 2
                                     for i in range(3)))
                print(f"  t={t:5.1f}s | "
                      f"EE: ({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}) | "
                      f"target dist: {dist:.4f}m")

            time.sleep(1.0 / 240.0)

    except (KeyboardInterrupt, p.error):
        print("\n  Simulation ended.")
    finally:
        stop_event.set()
        if HAS_MPL:
            plt.close("all")
        p.disconnect()
        print("  PyBullet disconnected. Bye!")


if __name__ == "__main__":
    main()
