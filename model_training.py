import os

os.system("pip install -q ultralytics supervision roboflow")

# Import ultralytics and check setup
import ultralytics
ultralytics.checks()

# Install roboflow
os.system("pip install roboflow")

from roboflow import Roboflow
rf = Roboflow(api_key="KdEogfVYcQLekR55rGHd")
project = rf.workspace("zara-hara").project("car-plate-k6xij")
version = project.version(22)
dataset = version.download("yolov11")
                

# Verify dataset download and check for data.yaml
print("Dataset downloaded to:", dataset.location)
data_yaml_path = os.path.join(dataset.location, "data.yaml")
if os.path.exists(data_yaml_path):
    print(f"data.yaml found at: {data_yaml_path}")
    # Train YOLO m
    os.system(f"yolo task=detect mode=train model=yolo11l.pt data={data_yaml_path} epochs=100 imgsz=640 plots=True")
else:
    print("data.yaml not found. Please check the dataset download.")
