from ultralytics import YOLO

model = YOLO("SSP_Fueldoor_rectangle_v1.pt")
print(model.names)
