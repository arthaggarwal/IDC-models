﻿# IDC-models

commands for the models

- exporting to ncnn: yolo export model=best.pt format=ncnn imgsz=640 simplify=True
- old - training: yolo task=detect mode=train model=yolov8n.pt data={data.path}/data.yaml epochs=40 imgsz=1080 batch=16 rect=True device=0
- new - training: yolo task=detect mode=train model=yolov8n.pt data={data.path}/data.yaml epochs=30 imgsz=1080 device=0
- also ensure that you update the path for the dataset in the data.yaml when downloading a new one
