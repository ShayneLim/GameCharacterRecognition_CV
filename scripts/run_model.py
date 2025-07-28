import torch
import cv2
import numpy as np
import mss
import time
from pathlib import Path

def load_yolov5_model(model_path):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"using {device}")
        if device == 'cuda':
            print(f"gpu: {torch.cuda.get_device_name(0)}")
        
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model = model.to(device)
        print(f"loaded {model_path}")
        return model
    except Exception as e:
        print(f"couldn't load model: {e}")
        return None

def capture_screen():
    # grab screenshot
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        
        # convert to opencv format
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

def draw_detections(img, results):
    # colors for each class
    class_colors = {
        'hero': (0, 255, 0),      # green
        'minion': (255, 0, 0),    # blue
        'tower': (0, 0, 255)      # red
    }
    default_color = (0, 255, 255)  # yellow
    
    detections = results.pandas().xyxy[0]
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = detection['name']
        
        color = class_colors.get(cls.lower(), default_color)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f'{cls} {conf:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        
        cv2.rectangle(img, (x1, label_y - label_size[1] - 4), (x1 + label_size[0], label_y), color, -1)
        cv2.putText(img, label, (x1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img

def main():
    MODEL_PATH = 'scripts\\weights\\best.pt'
    CONFIDENCE_THRESHOLD = 0.3
    WINDOW_NAME = 'YOLOv5 Screen Detection'
    
    if not Path(MODEL_PATH).exists():
        print(f"can't find {MODEL_PATH}")
        return
    
    print("loading yolo...")
    model = load_yolov5_model(MODEL_PATH)
    if model is None:
        return
    
    model.conf = CONFIDENCE_THRESHOLD
    
    print(f"starting detection...")
    print("q = quit, p = pause")
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    paused = False
    
    try:
        while True:
            start_time = time.time()
            
            if not paused:
                screen = capture_screen()
                results = model(screen)
                annotated_screen = draw_detections(screen.copy(), results)
                
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_screen, f'FPS: {fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                status = "PAUSED" if paused else "RUNNING"
                cv2.putText(annotated_screen, f'Status: {status}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                device = "GPU" if torch.cuda.is_available() else "CPU"
                cv2.putText(annotated_screen, f'Device: {device}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow(WINDOW_NAME, annotated_screen)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                
    except KeyboardInterrupt:
        print("\nstopped")
    except Exception as e:
        print(f"error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("done")

if __name__ == "__main__":
    main()