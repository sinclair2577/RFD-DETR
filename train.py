from ultralytics import RTDETR

model = RTDETR('ultralytics/cfg/models/rfd-detr.yaml') # 

model.train(data='./dataset/rfd/data.yaml',
                pretrained=False,
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                workers=16,
                device='0',
                patience=100,
                deterministic=False,
                project='runs/train',
                name='rfd-detr',
                exist_ok=True,
                )