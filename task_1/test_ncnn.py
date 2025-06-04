import os
import sys
import argparse
import glob
import time
import serial
import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
# Example usage from command line:
# python detect_and_communicate.py --model=path/to/your/best.pt --source=usb0 --resolution=1280x720

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float) # Ensure threshold is float
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print(f'ERROR: Model path "{model_path}" is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

source_type = None # Initialize source_type
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list: # Use .lower() for case-insensitive check
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'ERROR: File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source.lower(): # Use .lower() for case-insensitive check
    source_type = 'usb'
    try:
        usb_idx = int(img_source[3:])
    except ValueError:
        print(f'ERROR: Invalid USB camera index in "{img_source}". Expected format like "usb0", "usb1".')
        sys.exit(0)
elif 'picamera' in img_source.lower():
    source_type = 'picamera'
    try:
        picam_idx = int(img_source[8:])
    except ValueError:
        print(f'ERROR: Invalid Picamera index in "{img_source}". Expected format like "picamera0".')
        sys.exit(0)
else:
    print(f'ERROR: Input "{img_source}" is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
resW, resH = None, None
if user_res:
    resize = True
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    except ValueError:
        print(f'ERROR: Invalid resolution format "{user_res}". Expected format like "1280x720".')
        sys.exit(0)

# Check if recording is valid and set up recording
recorder = None
if record:
    if source_type not in ['video','usb', 'picamera']: # Picamera also supported now
        print('ERROR: Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('ERROR: Please specify resolution to record video at (--resolution argument).')
        sys.exit(0)

    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    try:
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))
        if not recorder.isOpened():
            raise IOError("Could not open video writer.")
        print(f"Recording output to {record_name} at {resW}x{resH}@{record_fps} FPS.")
    except Exception as e:
        print(f"ERROR: Failed to set up video recording: {e}. Recording will be disabled.")
        record = False

# Load or initialize image source
cap = None
imgs_list = []
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    filelist = glob.glob(os.path.join(img_source, '*')) # Use os.path.join for cross-platform paths
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext.lower() in img_ext_list:
            imgs_list.append(file)
    if not imgs_list:
        print(f"WARNING: No image files found in folder: {img_source}")
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {cap_arg}")
        sys.exit(0)
    # Set camera or video resolution if specified by user
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        # Verify resolution (sometimes cameras don't support requested resolution)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != resW or actual_height != resH:
            print(f"WARNING: Camera could not set resolution to {resW}x{resH}. Actual: {actual_width}x{actual_height}")

elif source_type == 'picamera':
    try:
        from picamera2 import Picamera2
        cap = Picamera2()
        # Ensure resolution is set if using picamera
        if not user_res:
             print("ERROR: Picamera requires --resolution argument.")
             sys.exit(0)
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cap.start()
        print(f"Picamera2 initialized at {resW}x{resH}.")
    except ImportError:
        print("ERROR: picamera2 library not found. Please install it for Picamera source.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to initialize Picamera2: {e}. Exiting.")
        sys.exit(0)


# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# --- SIMPLIFIED SERIAL COMMUNICATION SETUP ---
ser = None # Initialize ser to None
if __name__ == '__main__':
    try:
        # Open serial port with a short timeout
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)
        # IMPORTANT: Clear the input buffer immediately after opening
        # This gets rid of any leftover data from a previous session or Arduino bootup messages
        ser.reset_input_buffer()
        print("Serial port /dev/ttyUSB0 opened and input buffer cleared.")

    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port /dev/ttyUSB0: {e}. Serial communication will be disabled.")
        ser = None
    except Exception as e: # Catch other potential errors during serial setup
        print(f"An unexpected error occurred during serial setup: {e}")
        ser = None
# --- END SIMPLIFIED SERIAL COMMUNICATION SETUP ---

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    frame = None
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break # Use break instead of sys.exit(0) for cleaner cleanup
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
        if frame is None:
            print(f"ERROR: Could not read image file: {img_filename}")
            continue # Skip to next iteration if image fails to load

    elif source_type == 'video' or source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Reached end of the video file or unable to read frames from camera. Exiting program.')
            break

    elif source_type == 'picamera':
        try:
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('ERROR: Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
                break
        except Exception as e:
            print(f"ERROR: Picamera frame capture failed: {e}. Exiting.")
            break

    # If frame is still None here, something went wrong, skip to next iteration
    if frame is None:
        print("WARNING: Frame is None. Skipping inference for this iteration.")
        continue

    # Resize frame to desired display resolution if requested
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get center x-position of bounding box
        x_center = (xmin + xmax) // 2

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Print x-center position (useful for debugging)
        # print(f'Detected {classname} at x position: {x_center}')

        # Check if the detected object is a hotdog and determine its position
        if classname.lower() == 'objects':
            # Dynamically determine the resolution from the current frame
            current_frame_width = frame.shape[1]
            # current_frame_height = frame.shape[0] # Not directly used for X position

            # Calculate the one-third divisions based on the current frame width
            left_third_boundary = current_frame_width / 3
            right_third_boundary = (2 * current_frame_width) / 3

            position = 'center' # Default to center
            if x_center < left_third_boundary:
                position = 'left'
            elif x_center > right_third_boundary:
                position = 'right'

            print(f'DEBUG: Hotdog x_center: {x_center}, Python determined position: {position}')
            print(f'DEBUG: Frame Width: {current_frame_width}, Left Boundary: {left_third_boundary:.2f}, Right Boundary: {right_third_boundary:.2f}')


            # Send position over serial if serial communication is active
            if ser and ser.is_open:
                try:
                    message_to_send = f"at {position}\n".encode('utf-8')
                    ser.write(message_to_send)
                    print(f"DEBUG: Python actually sending: '{message_to_send.decode().strip()}'") # Print what's sent

                    # Read response from serial, but don't block indefinitely
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8').rstrip()
                        if line: # Only print if there's actual content
                            print(f"DEBUG: Serial response from Arduino: '{line}'")
                        else:
                            print("DEBUG: No immediate serial response received (empty line).")
                    else:
                        print("DEBUG: No immediate serial response received (no data in buffer).")
                except serial.SerialException as e:
                    print(f"ERROR: Serial communication error during loop: {e}. Disabling serial.")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None # Disable serial if an error occurs
                except Exception as e:
                    print(f"ERROR: An unexpected error occurred during serial send/read: {e}")
                    if ser and ser.is_open:
                        ser.close()
                    ser = None
            else:
                print("DEBUG: Serial communication not active (serial port not open or error occurred).")


        # Draw bounding box if confidence threshold is high enough
        if conf > min_thresh: # Use min_thresh from argparse
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)
    if record and recorder: # Check if recorder object is valid
        recorder.write(frame)

    # If inferencing on individual images, wait for user keypress. Otherwise, wait 5ms.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    else: # Video, USB, Picamera
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0) # Remove oldest
    frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# --- CLEAN UP ---
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb'] and cap:
    cap.release()
elif source_type == 'picamera' and cap:
    cap.stop()
if record and recorder:
    recorder.release()
    print(f"Recording saved to {record_name}.")

if ser and ser.is_open: # Ensure serial port is closed if it was opened
    ser.close()
    print("Serial port closed.")
cv2.destroyAllWindows()
