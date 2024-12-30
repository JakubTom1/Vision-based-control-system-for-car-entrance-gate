from ultralytics import YOLO
model = YOLO('yolo11n.pt')
result = model.train(data="<<exact_localization>>/data_char.yaml", epochs=11)