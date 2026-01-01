#!/usr/bin/env python3
# updated_yolo_detect_arducam_with_flask.py
"""
YOLO + TF-Luna integration + Flask Web Dashboard:
- LiDAR parsing with checksum & signed temp
- LiDAR smoothing for background values
- LiDAR used ONLY when a drone is detected and reading is fresh (post-frame)
- Flask backend running on port 5000 to serve status JSON and dashboard HTML
"""

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


from flask import Flask, jsonify, send_from_directory, Response

# ========================= FLASK BACKEND START =========================
app = Flask(__name__)

# shared values to expose to frontend
shared_data = {
    "drone_count": 0,
    "distance_cm": None
}
latest_frame = None
frame_lock = threading.Lock()
@app.route("/status")
def status():
    # return JSON with detection + distance
    return jsonify(shared_data)

@app.route("/dashboard")
def serve_dashboard():
    # serve dashboard HTML (expects dashboard.html in same folder)
    return send_from_directory(".", "dashboard.html")

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

#  MJPEG live video streaming endpoint
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                ret, jpeg = cv2.imencode(".jpg", latest_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   frame_bytes + b"\r\n")

            time.sleep(0.03)  # ~30 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def start_flask():
    # Run flask in a thread
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# start flask thread so it doesn’t block camera inference
flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()
# ========================== FLASK BACKEND END ==========================


# ----------------- TF-Luna background reader (corrected) -----------------
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
        # latest = (smooth_dist_cm, strength, temp_c, timestamp)
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
        # checksum is low 8 bits of sum of bytes 0..7
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
                        # parse frames
                        while len(self.buf) >= self.PACKET_SIZE:
                            # ensure header at start
                            if self.buf[0] != self.HEADER0 or self.buf[1] != self.HEADER1:
                                # drop until possible header
                                del self.buf[0]
                                continue
                            frame = self.buf[:self.PACKET_SIZE]
                            if not self._checksum_ok(frame):
                                # bad packet -> drop first byte and resync
                                del self.buf[0]
                                continue
                            # parse fields
                            dist_cm = frame[2] | (frame[3] << 8)         # unsigned
                            strength = frame[4] | (frame[5] << 8)
                            temp_raw = frame[6] | (frame[7] << 8)
                            # signed 16-bit conversion for temp
                            if temp_raw >= 32768:
                                temp_raw -= 65536
                            temp_c = temp_raw / 8.0

                            ts = time.time()
                            # smoothing (moving average on cm) for background/latest
                            self._smooth_window.append(dist_cm)
                            smooth_cm = int(sum(self._smooth_window) / len(self._smooth_window))

                            with self.lock:
                                self.latest = (smooth_cm, strength, temp_c, ts)

                            # consume packet
                            del self.buf[:self.PACKET_SIZE]
                else:
                    # no bytes waiting
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
        # signal loop to exit
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        # wait briefly for thread to exit
        try:
            if self.is_alive():
                self.join(timeout=1.0)
        except Exception:
            pass

# ----------------- helper to find likely serial port -----------------
def find_serial_port():
    # look for typical USB/ACM devices first
    patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/serial*', '/dev/ttyS*', '/dev/ttyAMA*']
    for pat in patterns:
        lst = glob.glob(pat)
        if lst:
            # return first candidate
            for p in lst:
                if os.path.exists(p):
                    return p
    # fallback to the small known list
    for p in ["/dev/serial0", "/dev/ttyUSB0", "/dev/ttyS0", "/dev/ttyAMA0"]:
        if os.path.exists(p):
            return p
    return None

# ----------------- Helper: fresh lidar samples after a frame -----------------
def get_fresh_lidar(lidar_reader, frame_ts, max_wait=0.25, n_avg=3, required_interval=0.01):
    """
    Wait for up to max_wait seconds to collect n_avg TF-Luna snapshots that are newer than frame_ts.
    Returns (avg_dist_cm, avg_strength, avg_temp_c, last_ts) or None if no fresh samples found.
    """
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
        # accept only samples strictly newer than frame_ts and newer than previous sample
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

# ----------------- Argument Parser -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source, e.g. "picamera0", "arducam", "usb0", "test.mp4", folder, or image', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480")', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution.', action='store_true')
parser.add_argument('--port', help='Serial port for TF-Luna (optional). If not provided, the script will try to auto-detect.', default=None)
parser.add_argument('--smoothing', help='Moving average window size for TF-Luna distance smoothing (default=5)', type=int, default=5)
parser.add_argument('--center-only', help='Only use LiDAR distance if bbox center is near image center (fraction of width/height, default=0.25)', type=float, default=0.25)
parser.add_argument('--lidar-min-strength', help='Minimum strength to accept LiDAR reading as reliable (default=100)', type=int, default=100)
parser.add_argument('--lidar-max-wait', help='Max wait (s) for fresh LiDAR samples after a frame (default=0.25)', type=float, default=0.25)
parser.add_argument('--lidar-navg', help='Number of fresh LiDAR samples to average (default=3)', type=int, default=3)
args = parser.parse_args()

# ----------------- Model Setup -----------------
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

if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

# ----------------- Source type detection -----------------
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif 'arducam' in img_source:
    source_type = 'arducam'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

resize = False
if user_res:
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        resize = True
    except Exception as e:
        print(f'Invalid resolution format: {user_res}. Expected WxH like 640x480.')
        sys.exit(0)

if record:
    if source_type not in ['video', 'usb', 'arducam']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# ----------------- Open source and LiDAR -----------------
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
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video':
        cap_arg = img_source
    else:
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()
elif source_type == 'arducam':
    # Auto-detect the first available /dev/video device for Arducam
    arducam_idx = None
    for i in range(4):  # check /dev/video0 → /dev/video3
        test_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if test_cap.isOpened():
            arducam_idx = i
            test_cap.release()
            break
    if arducam_idx is None:
        print('No Arducam device found. Exiting.')
        sys.exit(0)
    print(f"Using Arducam at /dev/video{arducam_idx}")
    cap = cv2.VideoCapture(arducam_idx, cv2.CAP_V4L2)

    # Try to grab first frame to confirm camera works
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera failed at init")
        sys.exit(0)
    print("Camera OK, now opening LiDAR...")

# If camera present, initialize LiDAR reader
camera_opened = source_type in ['video', 'usb', 'picamera', 'arducam']
if camera_opened:
    if source_type == 'picamera':
        test_frame = cap.capture_array()
        if test_frame is None:
            print("Unable to read frames from the Picamera. Exiting program.")
            if source_type in ['video', 'usb', 'arducam']:
                cap.release()
            elif source_type == 'picamera':
                cap.stop()
            sys.exit(0)

    # find serial port and start reader (works for /dev/serial0, /dev/ttyUSB0, etc.)
    port = user_port if user_port else find_serial_port()
    if port is None:
        print("[LiDAR] No serial port found (checked common paths). LiDAR will remain disabled.")
    else:
        print(f"[LiDAR] Starting reader on {port} (baud {115200}).")
        lidar_reader = TFLunaReader(port, 115200, smoothing_window=smoothing_window)
        if lidar_reader.ser:
            lidar_reader.start()
        else:
            lidar_reader = None

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# ----------------- Inference Loop -----------------
try:
    while True:
        t_start = time.perf_counter()

        # read frame from whichever source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                sys.exit(0)
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
            # timestamp frame right after capture/read
            frame_ts = time.time()
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
            frame_ts = time.time()
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the USB camera. Exiting program. ')
                break
            frame_ts = time.time()
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is None:
                print('Unable to read frames from the Picamera. Exiting program. ')
                break
            frame_ts = time.time()
        elif source_type == 'arducam':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the Arducam. Exiting program. ')
                break
            frame_ts = time.time()

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # run yolov8 inference (Ultralytics)
        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = 0

        height, width = frame.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0

        

        shared_data["drone_count"] = 0
        shared_data["distance_cm"] = None


        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = float(detections[i].conf.item())

            if conf >= float(min_thresh):
                object_count += 1
                
                
                if classname.lower() == "drone":
                    shared_data["drone_count"] += 1

                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf*100)}%'
                cv2.putText(frame, label, (xmin, max(12, ymin-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                # ------------- LiDAR Integration -------------
                # Only use LiDAR when:
                # - lidar initialized
                # - the detected class is 'drone' (case-insensitive)
                # - and either there's only 1 detection OR the bbox center is close to image center (likely aligned with single-point LiDAR)
                use_lidar = False
                if lidar_reader and classname.lower() == "drone":
                    # if only one detection, use it
                    if len(detections) == 1:
                        use_lidar = True
                    else:
                        # check if this bbox center is near image center
                        bx_center = (xmin + xmax) / 2.0
                        by_center = (ymin + ymax) / 2.0
                        dx = abs(bx_center - center_x)
                        dy = abs(by_center - center_y)
                        # fraction threshold relative to width/height
                        if (dx <= center_only_frac * width) and (dy <= center_only_frac * height):
                            use_lidar = True

                if use_lidar:
                    # request fresh lidar samples taken after this frame timestamp
                    fresh = get_fresh_lidar(lidar_reader, frame_ts, max_wait=lidar_max_wait, n_avg=lidar_navg, required_interval=0.01)
                    if fresh is not None:
                        dist_cm, strength, temp_c, ts = fresh
                        # reliability check
                        if strength < lidar_min_strength:
                            # treat as unreliable
                            cv2.putText(frame, "Dist: unreliable", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        else:
                            # Update shared distance with reliable reading
                            shared_data["distance_cm"] = dist_cm
                            
                            dist_m = dist_cm / 100.0
                            print(f"[{time.strftime('%H:%M:%S')}] Drone detected: {dist_cm} cm ({dist_m:.2f} m), strength={strength}, temp={temp_c:.1f}°C (lidar_ts={ts:.3f})")
                            cv2.putText(frame, f"Dist: {dist_cm} cm", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        # no fresh LiDAR reading — show placeholder (and avoid showing stale values)
                        cv2.putText(frame, "Dist: --", (xmin, max(12, ymin-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)
                else:
                    # intentionally avoid showing LiDAR data if ambiguous
                    pass

        # overlays
        if source_type in ['video', 'usb', 'picamera', 'arducam']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.putText(frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        with frame_lock:
            latest_frame = frame.copy()
        if os.environ.get("DISPLAY"):
            cv2.imshow('YOLO detection results', frame)
    	 

        if record:
            recorder.write(frame)

        if source_type in ['image','folder']:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', frame)

        t_stop = time.perf_counter()
        frame_rate_calc = float(1.0 / max(1e-6, (t_stop - t_start)))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        avg_frame_rate = float(np.mean(frame_rate_buffer))

finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ['video', 'usb', 'arducam']:
        cap.release()
    elif source_type == 'picamera':
        try:
            cap.stop()
        except Exception:
            pass
    if record:
        recorder.release()
    if lidar_reader:
        lidar_reader.stop()
    cv2.destroyAllWindows()
