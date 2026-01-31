from ultralytics import YOLO
import cv2
from pathlib import Path

def segment_video(input_path: Path, output_path: Path):
    # Convert to absolute paths
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # Check if input video exists
    if not input_path.exists():
        print(f"ERROR: Video not found: {input_path}")
        return
    
    print("Loading model...")
    model_path = Path(__file__).parent.parent / "models" / "best.pt"
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    model = YOLO(str(model_path))

    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("ERROR: Could not open video file!")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        print(f"Processing frame {frame_num}/{total_frames}")

        result = model(frame)[0]
        annotated = result.plot()
        out.write(annotated)

        # Display the frame in a window
        cv2.imshow("Segmentation", annotated)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nDone! Saved to {output_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()