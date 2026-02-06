from pathlib import Path
import sys
import onnxruntime as ort

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video_seg import segment_video

if __name__ == "__main__":
    print(f"Using ONNX Runtime {ort.__version__}")
    
    input_video = project_root / "data" / "videos" / "test1.mp4"
    output_video = project_root / "outputs" / "videos" / "test1_seg.mp4"
    
    success = segment_video(input_video, output_video, show_display=True)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")