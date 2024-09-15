import os
from ultralytics import YOLO
from openvino.runtime import Core
import cv2
import numpy as np

# Paths
base_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron', 'Back')
yolov8_path = os.path.join(base_path, 'yolov8')
valid_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron')
existing_data_yaml = os.path.join(yolov8_path, 'config.yaml')

def validate_model():
    model = YOLO(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt'))
    model.val(data=existing_data_yaml)

def test_model():
    model = YOLO(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt'))
    test_images_path = os.path.join(base_path, 'test', 'images')
    model.predict(source=test_images_path, save=True, conf=0.25)

def convert_to_openvino():
    model = YOLO(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt'))
    model.export(format='openvino', dynamic=True, half=True)

def infer_with_openvino():
    ie = Core()
    model = ie.read_model(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best_openvino_model', 'best.xml'))
    compiled_model = ie.compile_model(model, "AUTO")
    print("OpenVINO model loaded and compiled. Ready for inference.")

def test_single_image(image_path):
    # Load the model
    model = YOLO(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt'))
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    
    # Perform inference
    results = model(image)
    
    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0]
            class_id = box.cls[0]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    output_path = os.path.join(os.path.dirname(image_path), "output_" + os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

def webcam_detection():
    # Load the model
    model = YOLO(os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt'))
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Perform inference
        results = model(frame)
        
        # Process results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = box.cls[0]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Class: {class_id}, Conf: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Webcam Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Test a single image")
    print("2. Run webcam detection")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        image_path = input("Enter the path to the image you want to test: ")
        test_single_image(image_path)
    elif choice == '2':
        webcam_detection()
    else:
        print("Invalid choice. Exiting.")