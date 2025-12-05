# app.py
import threading
import time
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------
# 설정
# --------------------------
MODEL_PATH = "best.pt"   # best.pt 파일 경로 (필요시 절대경로로 수정)
CAMERA_ID = 0            # 기본 카메라 장치 id (웹캠이 여러개면 1,2...으로 변경)
ALERT_THRESHOLD = 0.7    # 경고를 띄울 confidence 임계값 (0.0 ~ 1.0)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# --------------------------

app = Flask(__name__)

# 전역상태 (스레드간 공유)
latest_frame = None    # JPEG 인코딩된 바이트 (bytes)
latest_alert = False   # 현재 alert 상태
frame_lock = threading.Lock()

# 모델 로드 (비동기 환경에서 한 번만)
print("Loading YOLO model from:", MODEL_PATH)
model = YOLO(MODEL_PATH)  # ultralytics YOLO object

def detection_thread():
    global latest_frame, latest_alert

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open video device", CAMERA_ID)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # YOLO 추론 (ultralytics API)
        # NOTE: model(frame) 는 리스트-like 결과를 줌. 첫 결과를 사용.
        results = model(frame, verbose=False)  # returns list-like
        res = results[0]

        alert_flag = False

        # boxes: res.boxes, could be empty
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # numpy arrays
            try:
                xyxy = boxes.xyxy.cpu().numpy()      # Nx4
                confs = boxes.conf.cpu().numpy()     # N
                classes = boxes.cls.cpu().numpy()    # N
            except Exception:
                # fallback if tensors on CPU already or different API
                xyxy = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                classes = np.array(boxes.cls)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                conf_val = float(conf)
                if conf_val >= ALERT_THRESHOLD:
                    alert_flag = True

                # draw box and label
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                # box color: red if above threshold else green
                color = (0, 0, 255) if conf_val >= ALERT_THRESHOLD else (0, 255, 0)
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                label = f"{int(cls)}: {conf_val*100:.1f}%"
                cv2.putText(frame, label, (x1i, y1i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # If alert_flag set, overlay big ALERT text
        if alert_flag:
            h, w = frame.shape[:2]
            overlay_text = "!!! WARNING: HIGH CONFIDENCE DETECTION !!!"
            cv2.putText(frame, overlay_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 3, cv2.LINE_AA)

        # encode frame as JPEG to stream
        ret2, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret2:
            continue

        jpeg_bytes = jpeg.tobytes()

        with frame_lock:
            latest_frame = jpeg_bytes
            latest_alert = bool(alert_flag)

        # slight throttle to avoid 100% CPU if needed
        # time.sleep(0.01)

    # cap.release()  # unreachable in this loop

# Start detection thread
t = threading.Thread(target=detection_thread, daemon=True)
t.start()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

def mjpeg_generator():
    """Yield latest_frame as multipart/x-mixed-replace stream."""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                # placeholder black image while initializing
                blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                ret, placeholder = cv2.imencode('.jpg', blank)
                frame = placeholder.tobytes()
            else:
                frame = latest_frame

        # yield multipart frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 fps cap

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert')
def alert_status():
    """Return JSON indicating if alert is active."""
    with frame_lock:
        return jsonify({"alert": latest_alert})

if __name__ == '__main__':
    # Flask 디버그 끄고, 호스트 0.0.0.0 으로 열면 외부에서 접근 가능
    app.run(host='0.0.0.0', port=5000, threaded=True)
