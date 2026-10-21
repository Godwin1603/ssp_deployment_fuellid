from ultralytics import YOLO

model = YOLO("SSP_FUEL_LID_V7.pt")
print(model.names)
