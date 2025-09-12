from flask import Blueprint, render_template, Response, jsonify, request, render_template_string
import cv2
import os
import threading
import torch
import time
import numpy as np
from ultralytics import RTDETR
from collections import deque, defaultdict

cam2_bp = Blueprint('cam2', __name__)

# --- YOLO 모델 (사람 탐지 전용) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔍 Using device: {device}")
DEVICE_ARG = 0 if device == 'cuda' else 'cpu'

# 모델을 지연 로딩
model = None

def get_model():
    global model
    if model is None:
        try:
            model = RTDETR("rtdetr-l.pt")
            # model = YOLO("yolo11s.pt", task="detect")
        except Exception as e:
            print(f"Model loading error: {e}")
            return None
    return model

def infer(img, **kwargs):
    m = get_model()
    if m is None:
        return [type('MockResult', (), {'boxes': None, 'plot': lambda img=img: img})()]
    return m.predict(img, device=DEVICE_ARG, imgsz=416, classes=[0], conf=0.25, iou=0.45, max_det=100, 
                     half=True, augment=False, verbose=False **kwargs)

# --- 영상 경로 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "static", "videos", "buchoen_station_square_2_30s.mp4")

# --- 전역 변수: 구역별 객체 탐지 결과 (MAIN 카메라용이지만 API 호환성을 위해 유지) ---
zone_detections = {1: 0, 2: 0, 3: 0, 4: 0}
detection_lock = threading.Lock()

# --- 경보 임계값 (API 호환성을 위해 유지) ---
WARNING_THRESHOLD = 50
DANGER_THRESHOLD = 50

# --- CAM2 이벤트 감지 설정 ---
cam2_zone_coords = {
    "running_zone": [(960, 540), (2000, 540), (2000, 1500), (960, 1500)],
}

# --- 이벤트 감지 설정 ---
HISTORY_LEN = 12
SPEED_WINDOW = 3
TRAIL_LEN = 6
SPEED_THRESHOLD_GENERAL = 2
track_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
event_detections = {"all_event_persons": set(), "all_event_count": 0}

# --- CAM2 이벤트 감지 함수들 ---
def is_valid_person_detection(x1, y1, x2, y2, conf):
    """사람 감지 후처리 검증"""
    w, h = x2 - x1, y2 - y1
    if w > 180 or h > 180:
        return False
    return True

def create_roi_mask(frame, roi_polygon):
    """ROI 마스크 생성"""
    pts = np.array(roi_polygon, np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(255,255,0), thickness=2)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def compute_speed(track_history_person, fps):
    """실제 비율을 반영한 속도 계산"""
    if len(track_history_person) < SPEED_WINDOW + 1:
        return 0.0
    p0 = track_history_person[0]
    p1 = track_history_person[SPEED_WINDOW]
    dt = (p1[0] - p0[0]) / fps if fps > 0 else 0.0
    if dt <= 0:
        return 0.0
    # 픽셀 차이
    dx = p1[1] - p0[1]  # x방향 (가로)
    dy = p1[2] - p0[2]  # y방향 (세로)
    # 픽셀 → 실제 거리(mm)
    scale_x = 7.89   # 가로 1px = 약 7.89 mm 기준 : 시각보조 보도블럭 가로 길이 30cm가정
    scale_y = 44.6   # 세로 1px = 약 44.6 mm 기준 : 횡단보도 세로길이 0.5m * 4 + 사이 폭 0.5m * 3
    dx_mm = dx * scale_x
    dy_mm = dy * scale_y
    # 실제 거리 (mm 단위)
    dist_mm = (dx_mm**2 + dy_mm**2) ** 0.5
    # 속도 = 거리 / 시간 (m/s로 변환)
    return (dist_mm / 1000.0) / dt

def detect_speed_events(detections, track_history, fps):
    """속도 감지 함수"""
    events = []
    for detection in detections:
        person_id = detection['id']
        center = detection['center']
        zone = detection.get('zone', 'general')
        
        if person_id >= 0 and person_id in track_history:
            speed = compute_speed(track_history[person_id], fps)
            if speed > SPEED_THRESHOLD_GENERAL:
                event_detections["all_event_count"] += 1
                event_detections["all_event_persons"].add(person_id)
                events.append({
                    'type': 'speed_event',
                    'person_id': person_id,
                    'speed': speed,
                    'center': center,
                    'zone': zone
                })
    return events

def generate_frames(camera_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = 10  # 프레임 건너뛰기를 늘려서 속도 조절
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # CAM2 - 이벤트 감지 화면 (핵심 로직)
        display_frame = frame.copy()
        
        # ROI 영역 그리기
        for zone_name, coords in cam2_zone_coords.items():
            pts = np.array(coords, np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 3)  # 노란색
            cv2.putText(display_frame, "RUNNING ZONE", (coords[0][0], coords[0][1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 이벤트 감지 수행
        all_detections = []
        mask = create_roi_mask(frame.copy(), cam2_zone_coords["running_zone"])
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 모델 로딩 시도
        try:
            m = get_model()
            if m is not None:
                # 사람 감지 및 트래킹
                results = m.track(
                    roi_frame,
                    persist=True,
                    conf=0.40,
                    iou=0.10,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            else:
                # 모델이 없으면 기본 프레임 사용
                results = None
        except Exception as e:
            print(f"모델 추론 오류: {e}")
            results = None
            
        if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
            ids = results[0].boxes.id
            for i, (box, conf) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf)):
                x1, y1, x2, y2 = map(int, box)
                
                if is_valid_person_detection(x1, y1, x2, y2, conf):
                    tid = int(ids[i]) if ids is not None else -1
                    
                    # 추적 이력 업데이트
                    if tid >= 0:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        w_box, h_box = (x2 - x1), (y2 - y1)
                        track_history[tid].append((frame_count, cx, cy, w_box, h_box))
                    
                    # 감지 정보 저장
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detection = {
                        'id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'confidence': float(conf),
                        'zone': 'running_zone'
                    }
                    all_detections.append(detection)
        
        # 이벤트 감지 (모델이 있을 때만)
        events = detect_speed_events(all_detections, track_history, fps) if results is not None else []
        
        # YOLO 결과 시각화
        if results is not None and results[0].boxes is not None:
            display_frame = results[0].plot(
                img=display_frame,
                labels=True,
                conf=False,
                probs=False,
                line_width=3
            )
        
        # 이벤트 시각화
        for event in events:
            if event['type'] == 'speed_event':
                center = event['center']
                speed = event['speed']
                person_id = event['person_id']
                
                # 속도 이벤트 표시
                cv2.circle(display_frame, center, 22, (0, 0, 255), 3)
                cv2.putText(display_frame, f"SPEED: {speed:.1f}m/s", 
                           (center[0]-45, center[1]-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                # 궤적 그리기
                if person_id >= 0 and person_id in track_history:
                    if len(track_history[person_id]) >= TRAIL_LEN:
                        pts = [(hx, hy) for _, hx, hy, _, _ in list(track_history[person_id])[-TRAIL_LEN:]]
                        for j in range(1, len(pts)):
                            cv2.line(display_frame, pts[j-1], pts[j], (255, 255, 0), 2)
        
        img = display_frame
        
        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 프레임 간격 조절로 비디오 속도 조절
        time.sleep(0.05)  # 50ms 지연 추가

    cap.release()


# ------------------ Flask Routes ------------------
@cam2_bp.route('/')
def index():
    return render_template('cam2.html', zone_coords=cam2_zone_coords)

@cam2_bp.route('/camera/<camera_id>')
def camera_feed(camera_id):
    print(f"카메라 요청: {camera_id} (타입: {type(camera_id)})")
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 아래 API들은 기존 시스템과의 호환성을 위해 유지 (실제로는 CAM2에서 사용 안함)
@cam2_bp.route('/api/zone_detections')
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

@cam2_bp.route('/api/detection_stats')
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

@cam2_bp.route('/api/alerts')
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


# ==============================================================================
# --- 프레임 크기 캐시 및 API ---
_FRAME_WH = [None, None]

def _compute_frame_wh():
    if _FRAME_WH[0] and _FRAME_WH[1]:
        return _FRAME_WH[0], _FRAME_WH[1]
    try:
        # CAM2 영상으로 크기 획득
        cap = cv2.VideoCapture(video_path)
        ret, fr = cap.read()
        cap.release()
        if ret:
            H, W = fr.shape[:2]
            _FRAME_WH[0], _FRAME_WH[1] = W, H
            return W, H
    except:
        pass
    return 1920, 1080

@cam2_bp.get("/api/frame_wh")
def api_frame_wh():
    W, H = _compute_frame_wh()
    return jsonify({"width": int(W), "height": int(H)})

# --- R 키로 /roi 팝업 열기 ---
@cam2_bp.get("/hotkeys.js")
def hotkeys_js():
    js = r"""
(()=>{function openPicker(){
  try{
    const w=window.open('/cam2/roi','roi','width=1280,height=820,noopener,noreferrer');
    if(!w) return;
  }catch(e){}
}
window.addEventListener('keydown',(e)=>{
  if(e.repeat) return;
  if(e.key==='r'||e.key==='R') openPicker();
});})();
"""
    return Response(js, mimetype="application/javascript")

# --- HTML 응답에 hotkeys.js 자동 삽입 ---
@cam2_bp.after_request
def inject_hotkeys(resp):
    try:
        ctype = resp.headers.get('Content-Type','')
        if 'text/html' in ctype.lower():
            body = resp.get_data(as_text=True)
            if ('/cam2/hotkeys.js' not in body) and ('</body>' in body):
                body = body.replace('</body>', '<script src="/cam2/hotkeys.js"></script></body>')
                resp.set_data(body)
    except Exception as e:
        print('[inject_hotkeys] skip:', e)
    return resp

# --- ROI 클릭 팝업 페이지 ---
@cam2_bp.get("/roi")
def roi_page():
    return render_template('roi_piker.html')

@cam2_bp.post("/api/roi/<int:zid>")
def api_set_roi(zid:int):
    data = request.get_json(silent=True) or {}
    pts = data.get("points", [])
    if not isinstance(pts, list) or len(pts) < 4:
        return jsonify({"ok": False, "error": "at least 4 points required"}), 400

    # 프레임 크기
    W, H = _compute_frame_wh()
    def clamp(v, lo, hi):
        v = int(round(float(v)))
        return max(lo, min(hi, v))

    xy = []
    try:
        for p in pts:
            x = clamp(p["x"], 0, W-1)
            y = clamp(p["y"], 0, H-1)
            xy.append((x, y))
    except Exception:
        return jsonify({"ok": False, "error": "invalid point format"}), 400

    # CAM2 러닝존에 무조건 저장 (zid는 UI용)
    cam2_zone_coords["running_zone"] = xy

    # (선택) 메인 ZONE overlay도 함께 갱신하고 싶으면 아래 줄을 켜세요.
    # zone_coords[str(zid)] = xy

    return jsonify({"ok": True, "zone": zid, "num_points": len(xy)})

@cam2_bp.get("/api/events_total")
def api_events_total():
    return jsonify({
        "events_total": len(event_detections["all_event_persons"])
    })

# @cam2_bp.get("/api/current_tracking_count")
# def get_current_tracking_count():
#     # 추적 중인 사람 수 반환
#     return jsonify({
#         'tracking_count': len(event_detections["all_event_persons"])
#     })