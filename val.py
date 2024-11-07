from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('')
    model.val(data='',
              imgsz=640,
              batch=64,
              project='runs/val',
              name='exp',
              )