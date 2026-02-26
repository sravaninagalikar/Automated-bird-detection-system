!pip install roboflow ultralytics
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="sFy3XfyryQBBlswCvfXT")
project = rf.workspace("detection-bjrn3").project("dettection-tracking-and-healthcare-monitoring")
version = project.version(1)
dataset = version.download("yolov8")
from ultralytics import YOLO

# Load the pretrained YOLOv8m segmentation model
model = YOLO("yolov8m.pt")

# Train the model on your custom dataset
results = model.train(
    data=f"{dataset.location}/data.yaml",  # Path to the dataset YAML file
    epochs=50,                             # Set to 50 epochs
    imgsz=640,                             # Default image size
    batch=8,                               # Batch size (adjust based on GPU memory)
    name="yolov8m-detection",           # Name of the training run
    device="0",                             # Use GPU (set to "cpu" if no GPU is available)
    optimizer="Adam")
