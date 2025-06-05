import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (e.g., "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True,
                    help='Image source: image file, folder, video file, or USB camera index (e.g., "usb0")')
parser.add_argument('--thresh', default=0.5,
                    help='Minimum confidence threshold (e.g., 0.4)', type=float)
parser.add_argument('--resolution', default=None,
                    help='Display resolution WxH (e.g., 640x480)')
parser.add_argument('--record', action='store_true',
                    help='Record video to "demo1.avi" (requires --resolution)')
parser.add_argument('--show_gui', action='store_true',
                    help='Show OpenCV GUI window (optional)')

args = parser.parse_args()

# Parse arguments
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
show_gui = args.show_gui

# Validate model path
if not os.path.exists(model_path):
    print('ERROR: Invalid model path.')
    sys.exit(1)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Detect input type
img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
vid_exts = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_exts:
        source_type = 'image'
    elif ext.lower() in vid_exts:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print('Invalid source input.')
    sys.exit(1)

# Handle resolution
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print("Invalid resolution format. Use WxH (e.g., 640x480)")
        sys.exit(1)

# Handle recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supported for video or camera sources.')
        sys.exit(1)
    if not resize:
        print('Recording requires resolution to be set.')
        sys.exit(1)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Load image/video source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_exts]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# FPS tracker
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Inference loop
while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('End of video or camera stream.')
            break
    elif source_type == 'picamera':
        frame = cv2.cvtColor(cap.capture_array(), cv2.COLOR_BGRA2BGR)

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    gauze_count = 0
    bandage_count = 0
    antiseptic_cream_count = 0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classid = int(det.cls.item())
        conf = det.conf.item()
        label = labels[classid]

        def normalize_label(label): return label.lower().replace(' ', '').replace('_', '')

        if conf >= min_thresh:
            object_count += 1

            if normalize_label(label) == 'gauze':
                gauze_count += 1
            elif normalize_label(label) == 'bandages':
                bandage_count += 1
            elif normalize_label(label) == 'antisepticcream':
                antiseptic_cream_count += 1

            if show_gui:
                color = bbox_colors[classid % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                text = f"{label}: {int(conf * 100)}%"
                cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Log detection counts
    print(f'Gauze: {gauze_count}, Bandage: {bandage_count}, Antiseptic Cream: {antiseptic_cream_count}')

    # Show GUI
    if show_gui:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO Detection", frame)

    if record:
        recorder.write(frame)

    # Keyboard input
    key = cv2.waitKey(5) if show_gui else -1
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite("capture.png", frame)

    # FPS
    t_end = time.perf_counter()
    fps = 1.0 / (t_end - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
if show_gui:
    cv2.destroyAllWindows()
