# Team WameedhX in BlastHakathon

# Oil Leak Detection with YOLO Segmentation
This project uses YOLO nano segmentation (Ultralytics) to detect and highlight oil leaks in video footage.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_video.py
```

Press **Q** to quit early. Output saved to `outputs/videos/`.

## Project Structure

```
├── data/videos/       # Input videos
├── models/best.pt     # YOLO model weights
├── outputs/videos/    # Output videos
├── scripts/           # Run scripts
└── src/               # Source code
```

