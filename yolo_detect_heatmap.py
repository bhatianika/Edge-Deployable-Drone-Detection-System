#!/usr/bin/env python3
# updated_yolo_detect_with_heatmap.py
# Adds image-space heatmap accumulation + optional crop logging
# Based on user's final integration script. See original file for full context. :contentReference[oaicite:1]{index=1}

import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import serial
import threading
import csv

# ---- Paste TF-Luna / serial helper classes from original file ----
# (unchanged logic from uploaded file; shortened comment here)
class TFLunaReader(threading.Thread):
    PACKET_SIZE = 9
    HEADER0 = 0x59
    HEADER1 = 0x59

    def __init__(self, port, baud=115200, smoothing_window=5):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = False
        self.buf = bytearray()
        self._smooth_window = deque(maxlen=max(1, smoothing_window))
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=0)
            self.running = True
        except Exception as e:
            print(f"[LiDAR] Could not open serial {self.port}: {e}")
            self.ser = None
            self.running = False

    @staticmethod
    def _checksum_ok(frame9: bytearray) -> bool:
        return (sum(frame9[0:8]) & 0xFF) == frame9[8]

    def run(self):
        if not self.ser:
            return
        while self.running:
            try:
                n = self.ser.in_waiting
                if n:
                    data = self.ser.read(n)
                    if data:
                        self.buf.extend(data)
                        while len(self.buf) >= self.PACKET_SIZE:
                            if self.buf[0] != self.HEADER0 or self.buf[1] != self.HEADER1:
                                del self.buf[0]
                                continue
                            frame = self.buf[:self.PACKET_SIZE]
                            if not self._checksum_ok(frame):
                                del self.buf[0]
                                continue
                            dist_cm = frame[2] | (frame[3] << 8)
                            strength = frame[4] | (frame[5] << 8)
                            temp_raw = frame[6] | (frame[7] << 8)
                            if temp_raw >= 32768:
                                temp_raw -= 65536
                            temp_c = temp_raw / 8.0
                            ts = time.time()
                            self._smooth_window.append(dist_cm)
                            smooth_cm = int(sum(self._smooth_window) / len(self._smooth_window))
                            with self.lock:
                                self.latest = (smooth_cm, strength, temp_c, ts)
                            del self.buf[:self.PACKET_SIZE]
                else:
                    time.sleep(0.005)
            except Exception as e:
                print(f"[LiDAR] serial read error: {e}")
                self.running = False
                break

    def snapshot(self):
        with self.lock:
            if self.latest is None:
                return None
            return tuple(self.latest)

    def stop(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        try:
            if self.is_alive():
                self.join(timeout=1.0)
        except Exception:
            pass

# util to find serial port (same as original)
def find_serial_port():
    patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/serial*', '/dev/ttyS*', '/dev/ttyAMA*']
    for pat in patterns:
        lst = glob.glob(pat)
        if lst:
            for p in lst:
                if os.path.exists(p):
                    return p
    for p in ["/dev/serial0", "/dev/ttyUSB0", "/dev/ttyS0", "/dev/ttyAMA0"]:
        if os.path.exists(p):
            return p
    return None

def get_fresh_lidar(lidar_reader, frame_ts, max_wait=0.25, n_avg=3, required_interval=0.01):
    if lidar_reader is None:
        return None
    samples = []
    start = time.time()
    last_sample_ts = 0.0
    while time.time() - start <= max_wait and len(samples) < max(1, n_avg):
        snap = lidar_reader.snapshot()
        if snap is None:
            time.sleep(0.005)
            continue
        dist_cm, strength, temp_c, ts = snap
        if ts > frame_ts and ts > last_sample_ts + required_interval:
            samples.append((dist_cm, strength, temp_c, ts))
            last_sample_ts = ts
        else:
            time.sleep(0.005)
    if not samples:
        return None
    avg_dist = int(sum(s[0] for s in samples) / len(samples))
    avg_strength = int(sum(s[1] for s in samples) / len(samples))
    avg_temp = float(sum(s[2] for s in samples) / len(samples))
    last_ts = samples[-1][3]
    return (avg_dist, avg_strength, avg_temp, last_ts)

# ---- Heatmap accumulator class ----
class HeatmapAccumulator:
    """
    Accumulate detections into a heatmap grid (image-space).
    - accumulator resolution independent of display resolution (cheap).
    - uses gaussian kernel per detection center.
    """
    def __init__(self, grid_w=160, grid_h=120, sigma=8, decay=0.995):
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.sigma = float(sigma)
        self.decay = float(decay)
        self.acc = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        # precompute gaussian patch
        size = max(3, int(self.sigma * 4))
        gx = cv2.getGaussianKernel(size, self.sigma)
        self.patch = (gx @ gx.T).astype(np.float32)

    def decay_step(self):
        self.acc *= self.decay

    def add(self, bbox, conf=1.0, dist_m=None, max_dist=12.0, weight_by_dist=False):
        """
        bbox: (x1,y1,x2,y2) coordinates in full frame pixel coords
        conf: detection confidence (0..1)
        dist_m: optional distance in meters
        weight_by_dist: if True, multiplies weight by (1 - dist/max_dist)
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        # map center to grid coords
        gx = int((cx / float(self.full_w)) * self.grid_w)
        gy = int((cy / float(self.full_h)) * self.grid_h)
        # compute weight
        w = float(conf)
        if weight_by_dist and dist_m is not None:
            w *= max(0.0, 1.0 - (dist_m / max_dist))
        # add gaussian patch centered at (gy,gx)
        size = self.patch.shape[0]
        half = size // 2
        y0 = gy - half; x0 = gx - half
        y1p = y0 + size; x1p = x0 + size
        py0 = 0; px0 = 0
        py1 = size; px1 = size
        if y0 < 0:
            py0 = -y0; y0 = 0
        if x0 < 0:
            px0 = -x0; x0 = 0
        if y1p > self.grid_h:
            py1 = size - (y1p - self.grid_h); y1p = self.grid_h
        if x1p > self.grid_w:
            px1 = size - (x1p - self.grid_w); x1p = self.grid_w
        if py1 - py0 > 0 and px1 - px0 > 0:
            self.acc[y0:y1p, x0:x1p] += w * self.patch[py0:py1, px0:px1]

    def set_frame_size(self, full_w, full_h):
        # needed to map pixel coords -> grid coords
        self.full_w = int(full_w)
        self.full_h = int(full_h)

    def get_overlay(self, frame_bgr, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """
        frame_bgr: original frame in BGR (full resolution)
        draws heatmap overlay (returns BGR overlay)
        """
        if self.acc.max() <= 0:
            return frame_bgr
        # normalize to 0..255
        heat = self.acc.copy()
        heat = heat / (heat.max() + 1e-9)
        heat_img = (heat * 255).astype(np.uint8)
        # resize heatmap to frame size
        heat_up = cv2.resize(heat_img, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat_col = cv2.applyColorMap(heat_up, colormap)
        overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_col, alpha, 0)
        return overlay

    def reset(self):
        self.acc.fill(0.0)

# ---- Argument parser (extend original) ----
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source (arducam, usb0, video.mp4, folder, image)', required=True)
parser.add_argument('--thresh', help='Min detection confidence', default=0.5, type=float)
parser.add_argument('--resolution', help='Display resolution WxH', default=None)
parser.add_argument('--record', help='Record output', action='store_true')
parser.add_argument('--port', help='Serial port for TF-Luna', default=None)
parser.add_argument('--smoothing', help='LiDAR smoothing window', type=int, default=5)
parser.add_argument('--center-only', help='Use LiDAR only if bbox near center (fraction)', type=float, default=0.25)
parser.add_argument('--lidar-min-strength', help='LiDAR min strength to accept', type=int, default=100)
parser.add_argument('--lidar-max-wait', help='Max wait for fresh LiDAR samples after frame(s)', type=float, default=0.25)
parser.add_argument('--lidar-navg', help='Number of LiDAR samples to average', type=int, default=3)

# heatmap options
parser.add_argument('--heatmap', help='Enable image-space heatmap overlay', action='store_true')
parser.add_argument('--heat-res', help='Heat accumulator resolution: WxH (default 160x120)', default='160x120')
parser.add_argument('--heat-sigma', help='Heat gaussian sigma (grid px)', default=8, type=float)
parser.add_argument('--heat-decay', help='Heat temporal decay per loop (0..1)', default=0.995, type=float)
parser.add_argument('--heat-weight-distance', help='Weight heat by inverse distance if LiDAR present', action='store_true')
parser.add_argument('--log-crops', help='Save detection crops and CSV for offline Grad-CAM', action='store_true')
parser.add_argument('--log-dir', help='Directory to save crops and CSV', default='./detection_logs')
args = parser.parse_args()

# ---- Model and source setup (mostly same as original) ----
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
smoothing_window = args.smoothing
user_port = args.port
center_only_frac = args.center_only
lidar_min_strength = args.lidar_min_strength
lidar_max_wait = args.lidar_max_wait
lidar_navg = max(1, args.lidar_navg)

if not os.path.exists(model_path):
    print('ERROR: model path invalid:', model_path)
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print('File extension not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'; usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'; picam_idx = int(img_source[8:])
elif 'arducam' in img_source:
    source_type = 'arducam'
else:
    print('Invalid input source:', img_source); sys.exit(0)

resize = False
if user_res:
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        resize = True
    except Exception as e:
        print('Invalid resolution format:', user_res); sys.exit(0)

if record and source_type not in ['video', 'usb', 'arducam']:
    print('Recording requires camera/video source'); sys.exit(0)

# prepare record writer if requested and camera
if record:
    record_name = 'demo1.avi'; record_fps = 30
    writer = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))
else:
    writer = None

# open capture similar to original
lidar_reader = None
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(os.path.join(img_source, '*'))
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb', 'picamera', 'arducam']:
    if source_type == 'video':
        cap_arg = img_source
    elif source_type == 'usb':
        cap_arg = usb_idx
    else:
        cap_arg = None
    if source_type == 'video' or source_type == 'usb' or source_type == 'arducam':
        cap = cv2.VideoCapture(cap_arg if cap_arg is not None else 0)
        if user_res:
            cap.set(3, resW); cap.set(4, resH)
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
        cap.start()

# start lidar if camera-like source present
camera_opened = source_type in ['video', 'usb', 'picamera', 'arducam']
if camera_opened:
    port = user_port if user_port else find_serial_port()
    if port is None:
        print("[LiDAR] No serial port found; LiDAR disabled.")
    else:
        print(f"[LiDAR] Starting reader on {port}")
        lidar_reader = TFLunaReader(port, 115200, smoothing_window=smoothing_window)
        if lidar_reader.ser:
            lidar_reader.start()
        else:
            lidar_reader = None

# bbox colors (copied)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# ---- Prepare heatmap + logging if requested ----
heatmap_enabled = bool(args.heatmap)
if heatmap_enabled:
    wr, hr = map(int, args.heat_res.split('x'))
    heat_acc = HeatmapAccumulator(grid_w=wr, grid_h=hr, sigma=args.heat_sigma, decay=args.heat_decay)
else:
    heat_acc = None

log_crops = bool(args.log_crops)
log_dir = args.log_dir
if log_crops:
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'detections.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['timestamp','frame','crop','x1','y1','x2','y2','conf','dist_m'])

# ---- Main inference loop (adapted from user's final script) ----
avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

try:
    while True:
        t_start = time.perf_counter()

        # read frame
        if source_type in ['image','folder']:
            if img_count >= len(imgs_list):
                print('All images processed; exiting.')
                break
            img_filename = imgs_list[img_count]; frame = cv2.imread(img_filename); img_count += 1
            frame_ts = time.time()
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Video ended.'); break
            frame_ts = time.time()
        elif source_type == 'usb' or source_type == 'arducam':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Camera read failed.'); break
            frame_ts = time.time()
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is None:
                print('Picamera read failed.'); break
            frame_ts = time.time()

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        full_h, full_w = frame.shape[:2]
        # if heatmap active, set heatmap mapping
        if heat_acc is not None:
            heat_acc.set_frame_size(full_w, full_h)
            heat_acc.decay_step()

        # run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = 0

        center_x = full_w / 2.0; center_y = full_h / 2.0

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = float(detections[i].conf.item())

            if conf >= float(min_thresh):
                object_count += 1
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf*100)}%'
                cv2.putText(frame, label, (xmin, max(12, ymin-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                # ---- LiDAR usage logic (same as original) ----
                use_lidar = False
                if lidar_reader and classname.lower() == "drone":
                    if len(detections) == 1:
                        use_lidar = True
                    else:
                        bx_center = (xmin + xmax) / 2.0
                        by_center = (ymin + ymax) / 2.0
                        dx = abs(bx_center - center_x); dy = abs(by_center - center_y)
                        if (dx <= center_only_frac * full_w) and (dy <= center_only_frac * full_h):
                            use_lidar = True

                dist_m = None
                if use_lidar:
                    fresh = get_fresh_lidar(lidar_reader, frame_ts, max_wait=lidar_max_wait, n_avg=lidar_navg, required_interval=0.01)
                    if fresh is not None:
                        dist_cm, strength, temp_c, ts = fresh
                        if strength < lidar_min_strength:
                            cv2.putText(frame, "Dist: unreliable", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        else:
                            dist_m = dist_cm / 100.0
                            cv2.putText(frame, f"Dist: {dist_cm} cm", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Dist: --", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

                # ---- Heatmap accumulation: add detection ----
                if heat_acc is not None:
                    heat_acc.add((xmin, ymin, xmax, ymax), conf=conf, dist_m=dist_m, weight_by_dist=args.heat_weight_distance)

                # ---- Optional crop logging for offline Grad-CAM ----
                if log_crops:
                    ts_ms = int(time.time() * 1000)
                    crop = frame[ymin:ymax, xmin:xmax]
                    crop_name = f'crop_{ts_ms}.jpg'
                    frame_name = f'frame_{ts_ms}.jpg'
                    cv2.imwrite(os.path.join(log_dir, crop_name), crop)
                    cv2.imwrite(os.path.join(log_dir, frame_name), frame)
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ts_ms, frame_name, crop_name, xmin, ymin, xmax, ymax, f"{conf:.3f}", "" if dist_m is None else f"{dist_m:.3f}"])

        # overlays & display
        if heat_acc is not None:
            overlay = heat_acc.get_overlay(frame, alpha=0.45)
        else:
            overlay = frame

        if camera_opened:
            cv2.putText(overlay, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(overlay, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.imshow('YOLO detection results (with heatmap)', overlay)
        if writer is not None:
            writer.write(overlay)

        if source_type in ['image','folder']:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', overlay)

        # frame rate measurement
        t_stop = time.perf_counter()
        frame_rate_calc = float(1.0 / max(1e-6, (t_stop - t_start)))
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0); frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = float(np.mean(frame_rate_buffer))

finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    # cleanup same as original
    try:
        if source_type in ['video','usb','arducam'] and 'cap' in locals():
            cap.release()
        elif source_type == 'picamera' and 'cap' in locals():
            cap.stop()
    except Exception:
        pass
    if writer is not None:
        writer.release()
    if lidar_reader:
        lidar_reader.stop()
    cv2.destroyAllWindows()
