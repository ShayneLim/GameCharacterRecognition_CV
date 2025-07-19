Folder Structure 

CVProject/
├── video/                         # Contains original gameplay clips (e.g., lux.mp4, test videos, negatives.mp4)  
├── dataset/  
│   ├── images/train/              # Extracted frames (.jpg)  
│   └── labels/train/              # Bounding box labels in YOLO format (.txt)  
├── yolov5/                        # Cloned YOLOv5 repo  
├── scripts/  
│   ├── extract_frames.py          # For extracting frames of Lux (positive class)  
│   └── extract_negatives.py       # For extracting background/other champions (negative class)  
├── data.yaml                      # YOLOv5 dataset configuration file  
└── weights/  
    └── best.pt                    # Trained YOLOv5 model weights (not included in repo)

install dependencies: pip install -r yolov5/requirements.txt
