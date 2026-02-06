from ultralytics import YOLO
import cv2
from pathlib import Path
import onnxruntime as ort

def segment_video(input_path: Path, output_path: Path, show_display: bool = True):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    if not input_path.exists():
        print(f"ERROR: Video not found: {input_path}")
        return False
    
    print("Loading ONNX model with ONNX Runtime...")
    model_path = Path(__file__).parent.parent / "models" / "best.onnx"
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return False
    
    ort.set_default_logger_severity(3)
    model = YOLO(str(model_path), task='segment')

    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("ERROR: Could not open video file!")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
    print(f"Using ONNX Runtime provider: {ort.get_available_providers()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("ERROR: Could not create output video!")
        cap.release()
        return False

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % 10 == 0 or frame_num == 1:
                print(f"Processing frame {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")

            results = model(frame, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

            if show_display:
                cv2.imshow("Segmentation", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted by user")
                    break

        print(f"\nDone! Processed {frame_num} frames. Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        return False
    finally:
        cap.release()
        out.release()
        if show_display:
            cv2.destroyAllWindows()