from flask import Flask, render_template, Response, jsonify
import cv2
import os
import threading
import torch
import subprocess
import time
import numpy as np
from ultralytics import RTDETR

app = Flask(__name__)

zone_coords = {
    "1": [(380, 630),(555, 570),(660, 660),(500, 730)],
    "2": [(1030, 475),(1180, 440),(1280, 480),(1160, 540)],
    "3": [(1390, 400),(1475, 365),(1575, 390),(1480, 430)],
    "4": [(1630, 670),(1750, 590),(1910, 630),(1800, 770)]
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔍 Using device: {device}")
DEVICE_ARG = 0 if device == 'cuda' else 'cpu'
model = RTDETR("C:/Users/Administrator/Desktop/wtdc/Python_Project/rtdetr-l.pt")

def infer(img, **kwargs):
    return model.predict(img, device=DEVICE_ARG, **kwargs)

# streamlink로 최적 스트림 URL 얻기
def get_stream_url(youtube_url):
    cmd = [
        r"C:\Users\Administrator\anaconda3\envs\py39\Scripts\streamlink.exe",
        "--stream-url",
        youtube_url,
        "best"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("streamlink error:", result.stderr)
        return None

youtube_live_url = "https://www.youtube.com/live/rnXIjl_Rzy4?si=lRcDqlANkukTPBAx"
stream_url = get_stream_url(youtube_live_url)

# --- 영상 경로 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = stream_url

# --- 전역 변수: 구역별 객체 탐지 결과 ---
zone_detections = {1: 0, 2: 0, 3: 0, 4: 0}
detection_lock = threading.Lock()

# --- 경보 임계값 ---
WARNING_THRESHOLD = 20
DANGER_THRESHOLD = 25

def generate_frames(camera_id):
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("스트림 열기 실패, 5초 후 재시도")
            cap.release()
            time.sleep(5)
            continue
    
        frame_skip = 10
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # MAIN 화면
            if str(camera_id) == 'main':
                img = frame.copy()
                for zone_id_str, coords in zone_coords.items():
                    zone_id = int(zone_id_str)
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    
                    # pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
                    
                    roi_frame = frame[y1:y2, x1:x2]
                    
                    roi_results = infer(roi_frame, 
                                        imgsz=416,
                                        classes=[0], 
                                        conf=0.3, 
                                        iou=0.5,
                                        max_det=100, 
                                        half=True,
                                        augment=False
                                        )

                    boxes = roi_results[0].boxes
                    count = boxes.shape[0] if boxes is not None and boxes.shape[0] > 0 else 0
                
                    with detection_lock:
                        zone_detections[zone_id] = count
                        
                    try:
                        roi_img = roi_results[0].plot(labels=False, probs=False)
                    except:
                        roi_img = roi_frame
                    img[y1:y2, x1:x2] = roi_img
                    
            # Zone만 스트리밍
            else:
                camera_num = str(camera_id)
                if camera_num in zone_coords:
                    coords = zone_coords[camera_num]
                    xs = [p[0] for p in coords]
                    ys = [p[1] for p in coords]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                    cropped_frame = frame[y1:y2, x1:x2].copy()

                    poly_pts = np.array([(x - x1, y - y1) for x, y in coords], np.int32).reshape((-1, 1, 2))
                    mask = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_pts], 255)
                    masked_img = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
                    
                    results = infer(masked_img, 
                                    imgsz=416,
                                    classes=[0], 
                                    conf=0.3,
                                    iou=0.5,
                                    max_det=100, 
                                    half=True,
                                    augment=False
                                    )
                    
                    try:
                        img = results[0].plot(labels=False, probs=False)
                    except:
                        img = masked_img
                else:
                    img = frame.copy()

            _, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

# ------------------ Flask Routes ------------------
@app.route('/')
def index():
    return render_template('index.html', zone_coords=zone_coords)

@app.route('/camera/<camera_id>')
def camera_feed(camera_id):
    print(f"카메라 요청: {camera_id} (타입: {type(camera_id)})")
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/zone_detections')
def get_zone_detections():
    with detection_lock:
        return jsonify({
            'zones': zone_detections,
            'total_detections': sum(zone_detections.values()),
            'danger_zones': [zone_id for zone_id, count in zone_detections.items() if count >= DANGER_THRESHOLD],
            'thresholds': {
                'warning': WARNING_THRESHOLD,
                'danger': DANGER_THRESHOLD,
            }
        })

@app.route('/api/detection_stats')
def detection_stats():
    with detection_lock:
        return jsonify({
            'total_detections': sum(zone_detections.values()),
            'active_cameras': 5,  # MAIN + 4개 카메라
            'zone_detections': zone_detections,
            'thresholds': {
                'warning': WARNING_THRESHOLD,
                'danger': DANGER_THRESHOLD,
            }
        })

@app.route('/api/alerts')
def get_alerts():
    with detection_lock:
        alerts = []
        for zone_id, count in zone_detections.items():
            if count >= DANGER_THRESHOLD:
                alerts.append(f"🚨 ZONE {zone_id}: {count}명 탐지 - 위험 수준!")
            elif count >= WARNING_THRESHOLD:
                alerts.append(f"⚡ ZONE {zone_id}: {count}명 탐지 - 주의 필요")
        
        if not alerts:
            alerts.append("✅ 모든 구역 정상 상태")
            
        return jsonify({'alerts': alerts})

# ------------------ 앱 실행 ------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)