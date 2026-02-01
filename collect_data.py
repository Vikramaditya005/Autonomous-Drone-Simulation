import airsim
import time, os, random
import numpy as np
import cv2
from datetime import datetime

# Output directory
OUT_DIR = "data_park"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Connect to AirSim with retry ---
client = airsim.MultirotorClient()
for i in range(30):
    try:
        client.confirmConnection()
        print("Connected!")
        break
    except Exception as e:
        print("Waiting for AirSim server...", i, e)
        time.sleep(1)
else:
    raise RuntimeError("Could not connect to AirSim after 30s")

client.enableApiControl(True)
client.armDisarm(True)

def read_obs():
    # Get RGB image
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
    ])
    img_resp = responses[0]
    if len(img_resp.image_data_uint8) == 0:
        return None

    img1d = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(img_resp.height, img_resp.width, 3)
    img = cv2.resize(img, (128, 128))

    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity

    pos_vec = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
    vel_vec = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)

    return img, pos_vec, vel_vec

def run_episode(episode_idx, num_steps=200):
    print(f"Starting episode {episode_idx}")
    client.takeoffAsync().join()

    # Random goal in some box
    goal = np.array([
        random.uniform(-30, 30),
        random.uniform(-30, 30),
        -random.uniform(3, 10)
    ], dtype=np.float32)

    # Lists to store per-step data
    rgbs = []
    poses = []
    vels = []
    goals = []
    actions = []

    for step in range(num_steps):
        obs = read_obs()
        if obs is None:
            print("No image from sim, stopping episode")
            break

        img, pos_vec, vel_vec = obs

        # Simple "expert" controller: move straight toward goal
        dx = goal - pos_vec
        # velocity command proportional to error, clamped
        vel_cmd = np.clip(dx, -3.0, 3.0)  # (vx, vy, vz)

        duration = 0.3
        client.moveByVelocityAsync(
            float(vel_cmd[0]),
            float(vel_cmd[1]),
            float(vel_cmd[2]),
            duration
        ).join()

        # Store data for this step
        rgbs.append(img)
        poses.append(pos_vec)
        vels.append(vel_vec)
        goals.append(goal.copy())
        actions.append(vel_cmd)

    client.landAsync().join()

    if len(rgbs) == 0:
        print("No data collected in this episode, skipping save")
        return

    # Stack lists into arrays
    rgbs_arr = np.stack(rgbs, axis=0)            # (T, H, W, 3)
    poses_arr = np.stack(poses, axis=0)          # (T, 3)
    vels_arr = np.stack(vels, axis=0)            # (T, 3)
    goals_arr = np.stack(goals, axis=0)          # (T, 3)
    actions_arr = np.stack(actions, axis=0)      # (T, 3)

    # Build filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(OUT_DIR, f"episode_{episode_idx}_{ts}.npz")

    # Save everything as named arrays
    np.savez_compressed(
        fn,
        rgbs=rgbs_arr,
        poses=poses_arr,
        vels=vels_arr,
        goals=goals_arr,
        actions=actions_arr,
    )
    print("Saved episode to", fn, "with", rgbs_arr.shape[0], "steps")

if __name__ == "__main__":
    # Collect e.g. 10 episodes to start
    for e in range(10):
        run_episode(e, num_steps=200)

    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done collecting data.")
