import os
from ultralytics import YOLO
from openvino.runtime import Core
import shutil
import yaml
from PIL import Image
import os
import cv2
import numpy as np

# Define paths
base_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron', 'Back')
yolov8_path = os.path.join(base_path, 'yolov8')

train_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron')

# Path to existing data.yaml
existing_data_yaml = os.path.join(yolov8_path, 'config.yaml')

def train_model():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data=existing_data_yaml, epochs=100, imgsz=640)

def validate_model():
    model = YOLO(os.path.join(train_path, 'runs/detect/train10/weights/best.pt'))
    model.val(data=existing_data_yaml)

def test_model():
    model = YOLO(os.path.join(train_path, 'runs/detect/train10/weights/best.pt'))
    # Adjust the source path to match your test images location
    test_images_path = os.path.join(yolov8_path, 'test', 'images')
    model.predict(source=test_images_path, save=True, conf=0.25)

def convert_to_openvino():
    model = YOLO(os.path.join(train_path, 'runs/detect/train10/weights/best.pt'))
    model.export(format='openvino', dynamic=True, half=True)

def infer_with_openvino():
    ie = Core()
    model = ie.read_model(os.path.join(train_path, 'runs/detect/train10/weights/best_openvino_model/best.xml'))
    compiled_model = ie.compile_model(model, "AUTO")
    
    # Placeholder for inference logic
    # You'll need to implement the actual inference steps here
    print("OpenVINO model loaded and compiled. Ready for inference.")

if __name__ == "__main__":
    #train_model()
    validate_model()
    test_model()
    convert_to_openvino()
    infer_with_openvino()