from ultralytics import RTDETR

model = RTDETR('best.pt') # your weights path
model.val(data='./dataset/rfd/data.yaml',
          split='valid', 
          imgsz=640,
          batch=16,
          exist_ok=True,
          project='runs/val',
          name='rfd-detr',
          )