# Team WameedhX in BlastHakathon

# Oil Leak Detection with YOLO Segmentation
This project uses YOLO nano segmentation (Ultralytics) to detect and highlight oil leaks in video footage.

# Team WameedhX — BlastHakathon

Oil leak detection using YOLO segmentation.

Install:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python scripts/run_video.py
```

Place the ONNX model at `models/best.onnx`. Output is written to `outputs/videos/`.

Project structure:

```
├── data/videos/
├── models/
├── outputs/videos/
├── scripts/
└── src/
```

