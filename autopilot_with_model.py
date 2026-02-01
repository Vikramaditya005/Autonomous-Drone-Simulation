import os
import time
import math
import traceback
import numpy as np
import torch
import torchvision
import cv2
import airsim

CKPT_PATH = os.path.join("models", "ckpt_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEED = 2.0
SAMPLE_HZ = 5.0
SCORE_THRESH = 0.5
GLOBAL_TIMEOUT = 120.0
MOVE_TIMEOUT = 10.0
STEP_FRACTION = 0.18
LATERAL_MAX_M = 1.2
BACKOFF_DISTANCE = 1.0
ASCEND_ON_BACKOFF = 0.6
IMG_W, IMG_H = 128, 128
CLOSE_THRESH_M = 6.0
DEPTH_BOX_W = 20
DEPTH_BOX_H = 20

def build_perception_model(device):
    """Construct the PerceptionNet architecture used in training (if you want to use it)."""
    import torch.nn as nn
    import torchvision.models as models

    class PerceptionNet(nn.Module):
        def __init__(self, state_goal_dim=9, hidden=256, use_pretrained=False):
            super().__init__()
            backbone = models.resnet18(pretrained=use_pretrained)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

            self.fc1 = nn.Linear(self.feature_dim + state_goal_dim, hidden)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 3)

        def forward(self, img, state_goal):
            feat = self.backbone(img)
            x = torch.cat([feat, state_goal], dim=1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.out(x)

    model = PerceptionNet().to(device)
    return model

def load_perception_model(ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        print("Perception model file not found:", ckpt_path)
        return None

    print("Attempting to load perception model:", ckpt_path)
    try:
        model = build_perception_model(device)
    except Exception:
        print("Error building perception architecture.")
        traceback.print_exc()
        return None

    try:
        chk = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print("Failed to torch.load checkpoint:", e)
        return None

    state_dict = None
    if isinstance(chk, dict):
        for k in ("model_state", "state_dict", "model", "model_state_dict"):
            if k in chk:
                state_dict = chk[k]
                break
        if state_dict is None:
            if any("." in str(k) for k in chk.keys()):
                state_dict = chk

    if state_dict is None:
        print("Perception checkpoint found but not loaded to model automatically (architecture must match).")
        return None

    try:
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("Model loaded successfully (strict=True).")
        return model
    except Exception as e:
        print("Strict load failed:", e)
        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("Loaded model with strict=False (partial load).")
            return model
        except Exception as e2:
            print("Failed to load model with strict=False:", e2)
            return None

def load_detector(device):
    print("Loading detector (Faster R-CNN)...")
    try:
        try:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        except Exception:
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        detector.to(device)
        detector.eval()
        print("Detector loaded.")
        return detector
    except Exception as e:
        print("Failed to load detector:", e)
        traceback.print_exc()
        return None
def get_front_camera_rgb(client, image_w=IMG_W, image_h=IMG_H):
    try:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if not responses or responses[0] is None:
            return None
        resp = responses[0]
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            return None
        img = img1d.reshape(resp.height, resp.width, 3)
        img = cv2.resize(img, (image_w, image_h))
        return img
    except Exception:
        traceback.print_exc()
        return None

def get_front_depth_meters(client, image_w=IMG_W, image_h=IMG_H):
    try:
        resp = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])[0]
        if resp is None:
            raise RuntimeError("No depth perspective response")
        if len(resp.image_data_float) > 0:
            arr = np.array(resp.image_data_float, dtype=np.float32)
            if arr.size == 0:
                return None
            depth = arr.reshape(resp.height, resp.width)
            depth = cv2.resize(depth, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            return depth
    except Exception:
        try:
            resp = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])[0]
            if resp is None:
                return None
            arr = np.array(resp.image_data_float, dtype=np.float32)
            if arr.size == 0:
                return None
            depth = arr.reshape(resp.height, resp.width)
            depth = cv2.resize(depth, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            return depth
        except Exception:
            return None

def depth_median_in_box(depth_map, cx, cy, box_w=DEPTH_BOX_W, box_h=DEPTH_BOX_H):
    if depth_map is None:
        return None
    h, w = depth_map.shape
    x1 = int(max(0, round(cx - box_w/2)))
    x2 = int(min(w, round(cx + box_w/2)))
    y1 = int(max(0, round(cy - box_h/2)))
    y2 = int(min(h, round(cy + box_h/2)))
    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))

def detect_boxes(detector, device, img_np, score_thresh=SCORE_THRESH):
    if detector is None:
        return []
    try:
        img = img_np.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1).to(device)
        with torch.no_grad():
            outputs = detector([img_t])
        if not outputs:
            return []
        out = outputs[0]
        boxes = out.get("boxes", None)
        scores = out.get("scores", None)
        if boxes is None or scores is None:
            return []
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        results = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s < score_thresh:
                continue
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            results.append((cx, cy, float(s)))
        return results
    except Exception:
        traceback.print_exc()
        return []
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def move_and_wait(client, tx, ty, tz, speed=1.0, timeout=MOVE_TIMEOUT):
    try:
        f = client.moveToPositionAsync(tx, ty, tz, speed)
    except Exception:
        try:
            f = client.moveToPositionAsync(tx, ty, tz, speed)
        except Exception:
            return False

    start = time.time()
    while True:
        st = client.getMultirotorState()
        pos = st.kinematics_estimated.position
        cur = (pos.x_val, pos.y_val, pos.z_val)
        if distance(cur, (tx, ty, tz)) < 0.4:
            try:
                if hasattr(f, "join"):
                    f.join()
            except Exception:
                pass
            return True
        if (time.time() - start) > timeout:
            try:
                if hasattr(f, "cancel"):
                    f.cancel()
            except Exception:
                pass
            return False
        time.sleep(0.08)

def burst_velocity_toward(client, tx, ty, tz, duration=0.6, speed=1.2):
    try:
        st = client.getMultirotorState()
        pos = st.kinematics_estimated.position
        cur = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        target = np.array([tx, ty, tz], dtype=np.float32)
        dirv = target - cur
        norm = np.linalg.norm(dirv)
        if norm < 0.05:
            return True
        vel = (dirv / (norm + 1e-8)) * speed
        f = client.moveByVelocityAsync(float(vel[0]), float(vel[1]), float(vel[2]), duration)
        time.sleep(duration + 0.05)
        try:
            if hasattr(f, "join"):
                f.join()
        except Exception:
            pass
        return True
    except Exception:
        return False
def plan_avoidance_waypoint(cur_pos, world_goal, box_center_px, img_w=IMG_W, img_h=IMG_H):
    if box_center_px is None:
        tx = cur_pos[0] + (world_goal[0] - cur_pos[0]) * 0.5
        ty = cur_pos[1] + (world_goal[1] - cur_pos[1]) * 0.5
        tz = cur_pos[2] + (world_goal[2] - cur_pos[2]) * 0.5
        return (tx, ty, tz)

    cx, cy = box_center_px
    dx_px = (cx - img_w / 2.0)
    lateral_m = np.clip((dx_px / (img_w/2.0)) * LATERAL_MAX_M, -LATERAL_MAX_M, LATERAL_MAX_M)

    tx = cur_pos[0] + (world_goal[0] - cur_pos[0]) * 0.35
    ty = cur_pos[1] + lateral_m
    tz = cur_pos[2] + (world_goal[2] - cur_pos[2]) * 0.25

    tz = float(np.clip(tz, cur_pos[2] - 1.2, cur_pos[2] + 1.0))
    return (tx, ty, tz)

def main():
    print("Device:", DEVICE)
    client = airsim.MultirotorClient()
    client.confirmConnection()

    percep_model = None
    try:
        percep_model = load_perception_model(CKPT_PATH, DEVICE)
        if percep_model is None:
            print("Perception model not available; continuing with detector-only.")
    except Exception:
        print("Exception while loading perception model; continuing with detector-only.")
        traceback.print_exc()
        percep_model = None

    detector = None
    try:
        detector = load_detector(DEVICE)
    except Exception:
        print("Failed to create detector; continuing without detection.")
        detector = None

    print("Connected!")
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        t_fut = client.takeoffAsync()
        try:
            t_fut.join()
        except Exception:
            
            time.sleep(1.0)
        print("Taking off...")
    except Exception:
        print("Takeoff failed; exiting.")
        return

    st = client.getMultirotorState()
    p = st.kinematics_estimated.position
    start_pos = (p.x_val, p.y_val, p.z_val)
    world_goal = (start_pos[0] + 40.0, start_pos[1], start_pos[2])

    start_time = time.time()
    last_sample = 0.0
    stuck_since = None

    try:
        while True:
            now = time.time()
            if (now - start_time) > GLOBAL_TIMEOUT:
                print("Global stuck timeout exceeded — forcing safe landing.")
                break

            if (now - last_sample) < (1.0 / SAMPLE_HZ):
                time.sleep(0.02)
                continue
            last_sample = now

            img = get_front_camera_rgb(client, image_w=IMG_W, image_h=IMG_H)
            if img is None:
                print("No camera image available this frame.")
                continue

            boxes = []
            if detector is not None:
                try:
                    boxes = detect_boxes(detector, DEVICE, img, score_thresh=SCORE_THRESH)
                except Exception:
                    print("Detector failed on this frame.")
                    traceback.print_exc()
                    boxes = []

            depth_map = get_front_depth_meters(client, image_w=IMG_W, image_h=IMG_H)

            if boxes:
                best = max(boxes, key=lambda x: x[2])
                cx, cy, score = best
                dist_m = depth_median_in_box(depth_map, cx, cy, box_w=DEPTH_BOX_W, box_h=DEPTH_BOX_H)
                if dist_m is None:
                    is_block = True
                else:
                    is_block = (dist_m < CLOSE_THRESH_M)
                print(f"Detected {len(boxes)} object(s). Best box cx,cy,score=({round(cx,1)},{round(cy,1)},{round(score,3)})  depth_m={dist_m if dist_m is not None else 'N/A'}  blocking={is_block}")

                if is_block:
                    st = client.getMultirotorState()
                    pos = st.kinematics_estimated.position
                    cur_pos = (pos.x_val, pos.y_val, pos.z_val)
                    tx, ty, tz = plan_avoidance_waypoint(cur_pos, world_goal, (cx, cy), IMG_W, IMG_H)
                    print("Attempting avoidance waypoint -> ({:.2f}, {:.2f}, {:.2f})".format(tx, ty, tz))
                    ok = move_and_wait(client, tx, ty, tz, speed=SPEED, timeout=MOVE_TIMEOUT)
                    if not ok:
                        print("Avoidance move timed out; trying velocity-burst fallback...")
                        ok2 = burst_velocity_toward(client, tx, ty, tz, duration=0.6, speed=1.2)
                        if not ok2:
                            print("Avoidance burst failed; hovering and replanning...")
                            try:
                                hf = client.hoverAsync()
                                try:
                                    hf.join()
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            time.sleep(0.4)
                            if stuck_since is None:
                                stuck_since = time.time()
                            else:
                                if time.time() - stuck_since > 10.0:
                                    print("Stuck for too long during avoidance; aborting and landing.")
                                    break
                            continue
                        else:
                            stuck_since = None
                    else:
                        stuck_since = None
                        continue
                else:
                    pass

            st = client.getMultirotorState()
            pos = st.kinematics_estimated.position
            cur_pos = (pos.x_val, pos.y_val, pos.z_val)
            tx = cur_pos[0] + (world_goal[0] - cur_pos[0]) * STEP_FRACTION
            ty = cur_pos[1] + (world_goal[1] - cur_pos[1]) * STEP_FRACTION
            tz = cur_pos[2] + (world_goal[2] - cur_pos[2]) * STEP_FRACTION
            print("Moving toward world goal -> ({:.2f}, {:.2f}, {:.2f})".format(tx, ty, tz))
            ok = move_and_wait(client, tx, ty, tz, speed=SPEED, timeout=MOVE_TIMEOUT)
            if not ok:
                print("Move timed out, hovering and retrying...")
                try:
                    hf = client.hoverAsync()
                    try:
                        hf.join()
                    except Exception:
                        pass
                except Exception:
                    pass
                if not burst_velocity_toward(client, tx, ty, tz, duration=0.6, speed=1.2):
                    if stuck_since is None:
                        stuck_since = time.time()
                    elif time.time() - stuck_since > 12.0:
                        print("Stuck for too long; aborting and landing.")
                        break
                else:
                    stuck_since = None
                time.sleep(0.3)
                continue
            else:
                stuck_since = None

            st = client.getMultirotorState()
            pos = st.kinematics_estimated.position
            cur = (pos.x_val, pos.y_val, pos.z_val)
            if distance(cur, world_goal) < 1.0:
                print("Reached world goal. Landing.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user, landing safely.")
    except Exception:
        print("Exception in main loop:")
        traceback.print_exc()
    finally:
        print("Finalizing: landing and releasing control.")
        try:
            lf = client.landAsync()
            try:
                lf.join()
            except Exception:
                time.sleep(1.0)
        except Exception:
            pass
        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
        print("Done.")

if __name__ == "__main__":
    main()
