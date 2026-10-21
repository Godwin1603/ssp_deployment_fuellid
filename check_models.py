from ultralytics import YOLO
import sys

def check_model(model_name):
    print(f"--- Checking {model_name} ---")
    try:
        model = YOLO(model_name)
        print("Classes:", model.names)
        # Also print task type (e.g. segment, detect) if available
        print("Task:", model.task)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

check_model("best (2)_cIRCLE.pt")
check_model("best (2)polygen.pt")
check_model("SSP_Fueldoor_rectangle_v1.pt")
