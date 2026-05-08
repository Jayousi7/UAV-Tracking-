import cv2 as cv
import sys
from inference import InferenceEngine


def run_system(tracker_type='ocsort'):
    model_path = "onnx_models/yolo26n_RGB.onnx"
    video_path = "vid.mp4"
    engine = InferenceEngine(model_path, tracker_type=tracker_type)

    cap = cv.VideoCapture(video_path)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        tracks = engine.process_frame(frame)

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Drone Surveillance System", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else 'ocsort'
    run_system(tracker_type=t)