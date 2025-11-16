"""
Configuration Helper Script
This script helps you find the optimal counting line position for your video.
Run this first to visually set the gate line position, then update main.py accordingly.
Also supports drawing a polygon ROI and saving it as ratios JSON.
"""
import sys
import json
import cv2
import numpy as np

# Load video (path from CLI or default)
video_path = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
cap = cv2.VideoCapture(0 if str(video_path) == '0' else video_path)

# Global variables for line adjustment
line_position = [0.5, 0.5]  # [x_ratio, y_ratio] for line position
line_type = "horizontal"  # "horizontal" or "vertical"
drawing = False
mode = 'line'               # 'line' or 'roi'
roi_points = []             # list of [xr, yr]

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to adjust line position or draw ROI polygon"""
    global line_position, line_type, drawing, mode, roi_points
    
    if mode == 'line':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            h, w = frame.shape[:2]
            if line_type == "horizontal":
                line_position[1] = y / h
            else:
                line_position[0] = x / w
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = frame.shape[:2]
            roi_points.append([x / w, y / h])

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video")
    exit()

h, w = frame.shape[:2]

print("=" * 60)
print("CONFIGURATION HELPER")
print("=" * 60)
print("Instructions:")
print("1. Press 'h' to set horizontal counting line")
print("2. Press 'v' to set vertical counting line")
print("3. Click and drag mouse to adjust line position")
print("4. Press 'r' to switch to ROI polygon mode (click to add points, 'u' undo, 'c' clear)")
print("5. Press 'l' to return to line mode")
print("6. Press 's' to save current configuration (line or ROI)")
print("7. Press 'q' to quit")
print("=" * 60)

cv2.namedWindow('Configure Gate/ROI', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Configure Gate/ROI', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame_copy = frame.copy()
    
    # Draw overlays
    if mode == 'line':
        if line_type == "horizontal":
            y_pos = int(h * line_position[1])
            cv2.line(frame_copy, (0, y_pos), (w, y_pos), (0, 255, 255), 3)
            cv2.putText(frame_copy, f'Horizontal Line at Y={y_pos} (ratio={line_position[1]:.2f})',
                        (10, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            x_pos = int(w * line_position[0])
            cv2.line(frame_copy, (x_pos, 0), (x_pos, h), (0, 255, 255), 3)
            cv2.putText(frame_copy, f'Vertical Line at X={x_pos} (ratio={line_position[0]:.2f})',
                        (x_pos + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        if len(roi_points) >= 1:
            pts_px = [(int(px * w), int(py * h)) for px, py in roi_points]
            if len(pts_px) >= 2:
                cv2.polylines(frame_copy, [np.array(pts_px, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)
            for (px, py) in pts_px:
                cv2.circle(frame_copy, (px, py), 3, (255, 0, 0), -1)
            cv2.putText(frame_copy, f'ROI points: {len(roi_points)} (click to add, u=undo, c=clear, s=save)', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display instructions
    cv2.putText(frame_copy, f'Line Type: {line_type.upper()}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_copy, "Press 'h' for horizontal, 'v' for vertical", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame_copy, "Click and drag to adjust position", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.imshow('Configure Gate/ROI', frame_copy)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        mode = 'line'
        line_type = "horizontal"
        print(f"Mode=line, Line type: {line_type}")
    elif key == ord('v'):
        mode = 'line'
        line_type = "vertical"
        print(f"Mode=line, Line type: {line_type}")
    elif key == ord('l'):
        mode = 'line'
        print("Switched to line mode")
    elif key == ord('r'):
        mode = 'roi'
        print("Switched to ROI mode. Click to add polygon points.")
    elif key == ord('u') and mode == 'roi' and roi_points:
        roi_points.pop()
    elif key == ord('c') and mode == 'roi':
        roi_points.clear()
    elif key == ord('s'):
        print("\n" + "=" * 60)
        print("CURRENT CONFIGURATION:")
        if mode == 'line':
            print(f"COUNTING_LINE_TYPE = \"{line_type}\"")
            if line_type == "horizontal":
                print(f"COUNTING_LINE_POS_RATIO = {line_position[1]:.3f}")
            else:
                print(f"COUNTING_LINE_POS_RATIO = {line_position[0]:.3f}")
            print("Use: --line-type and --line-pos in main.py")
        else:
            if len(roi_points) >= 3:
                out_path = 'roi.json'
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(roi_points, f, indent=2)
                print(f"Saved ROI polygon ratios to {out_path}. Pass it with: --roi {out_path}")
            else:
                print("ROI must have at least 3 points to save.")
        print("=" * 60 + "\n")

cap.release()
cv2.destroyAllWindows()
print("Configuration helper closed.")

