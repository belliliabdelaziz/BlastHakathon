from pathlib import Path
import sys

# Add project root to path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import AFTER setting path
from src.video_seg import segment_video

input_video = project_root / "data" / "videos" / "test2.mp4"
output_video = project_root / "outputs" / "videos" / "test2_seg.mp4"

segment_video(input_video, output_video)