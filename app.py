import asyncio
import base64
import json
import os
import time
import shutil
import cv2 as cv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from inference import InferenceEngine

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODELS = {
    'yolo26n_RGB': 'onnx_models/yolo26n_RGB.onnx',
    'yolo26s_RGB': 'onnx_models/yolo26s_RGB.onnx',
    'DT_RGB': 'onnx_models/DT_RGB.onnx',
    'yolo26n_IR': 'onnx_models/yolo26n_IR.onnx',
    'yolo26s_IR': 'onnx_models/yolo26s_IR.onnx',
    'DT_IR': 'onnx_models/DT_IR.onnx',
}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

video_path = None
engine = InferenceEngine(MODELS['yolo26n_RGB'], tracker_type='ocsort')


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    global video_path

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        return JSONResponse({"error": "Unsupported format. Use mp4/avi/mov/mkv/webm."}, status_code=400)

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv.VideoCapture(save_path)
    if not cap.isOpened():
        os.remove(save_path)
        return JSONResponse({"error": "Cannot read video file."}, status_code=400)

    info = {
        "filename": file.filename,
        "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv.CAP_PROP_FPS), 1),
        "frames": int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    video_path = save_path
    return JSONResponse({"status": "ok", "video": info})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global engine, video_path

    if video_path is None:
        await websocket.send_json({"type": "waiting", "message": "Upload a video to begin"})
        try:
            while video_path is None:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            await websocket.send_json({"type": "video_loaded", "message": "Video loaded"})
        except WebSocketDisconnect:
            return

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": "Cannot open video"})
        await websocket.close()
        return

    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_target = cap.get(cv.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps_target
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    paused = False
    current_model = 'yolo26n_RGB'
    current_tracker = 'ocsort'
    process_fps = 15

    await websocket.send_json({
        "type": "video_info",
        "width": frame_w,
        "height": frame_h,
        "fps": round(fps_target, 1),
        "frames": total_frames,
    })

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                data = json.loads(msg)

                if data.get("type") == "config":
                    new_model = data.get("model", current_model)
                    new_tracker = data.get("tracker", current_tracker)
                    new_fps = data.get("fps", process_fps)
                    process_fps = new_fps

                    if new_model != current_model and new_model in MODELS:
                        engine.switch_model(MODELS[new_model])
                        current_model = new_model

                    if new_tracker != current_tracker:
                        engine.switch_tracker(new_tracker)
                        current_tracker = new_tracker

                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

                elif data.get("type") == "pause":
                    paused = not paused

                elif data.get("type") == "restart":
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    engine.switch_tracker(current_tracker)
                    paused = False

                elif data.get("type") == "new_video":
                    cap.release()
                    cap = cv.VideoCapture(video_path)
                    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    fps_target = cap.get(cv.CAP_PROP_FPS) or 30
                    frame_delay = 1.0 / fps_target
                    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    engine.switch_tracker(current_tracker)
                    paused = False
                    await websocket.send_json({
                        "type": "video_info",
                        "width": frame_w, "height": frame_h,
                        "fps": round(fps_target, 1), "frames": total_frames,
                    })

            except asyncio.TimeoutError:
                pass

            if paused:
                await asyncio.sleep(0.05)
                continue

            t0 = time.perf_counter()

            if process_fps < fps_target:
                skip = int(round(fps_target / process_fps)) - 1
                for _ in range(skip):
                    ret, _ = cap.read()
                    if not ret: break

            ret, frame = cap.read()
            if not ret:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                engine.switch_tracker(current_tracker)
                continue

            tracks = engine.process_frame(frame)
            inference_time = time.perf_counter() - t0
            
            # Estimate actual stream FPS based on the sleep budget
            skip_count = int(round(fps_target / process_fps)) - 1 if process_fps < fps_target else 0
            loop_target_time = (1 + skip_count) * frame_delay
            actual_loop_time = max(loop_target_time, inference_time)
            stream_fps = 1.0 / max(actual_loop_time, 0.001)

            _, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer.tobytes()).decode('ascii')

            track_list = []
            for t in tracks:
                track_list.append({
                    "id": int(t[4]),
                    "x1": float(t[0]), "y1": float(t[1]),
                    "x2": float(t[2]), "y2": float(t[3]),
                })

            frame_num = int(cap.get(cv.CAP_PROP_POS_FRAMES))

            await websocket.send_json({
                "type": "frame",
                "frame": frame_b64,
                "tracks": track_list,
                "fps": round(stream_fps, 1),
                "frameNum": frame_num,
                "totalFrames": total_frames,
                "model": current_model,
                "tracker": current_tracker,
                "videoW": frame_w,
                "videoH": frame_h,
            })

            elapsed = time.perf_counter() - t0
            sleep_time = max(0, loop_target_time - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
