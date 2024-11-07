from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('')
    # model.load('yolov8n.pt') 
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=64,
                device='0',
                optimizer='SGD', 
                patience=0, 
                project='runs/train',
                name='exp',
                )