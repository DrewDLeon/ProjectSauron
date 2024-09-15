import os
import cv2
from ultralytics import YOLO

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
valid_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron')
model_path = os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt')
images_dir = os.path.join(valid_path, 'Back', 'Images', 'final_test')

def process_image(image_path):
    print(f"Starting image processing for {image_path}...")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return

    # Resize the image to 640x640
    image = cv2.resize(image, (640, 640))

    # Display the original image
    cv2.imshow("Original Image", image)
    cv2.waitKey(1)  # This will display the window

    # Load the model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)

    print("Performing inference...")
    # Perform inference
    results = model(image)

    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

    # Save the result
    output_path = os.path.join(script_dir, "output_" + os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

if __name__ == "__main__":
    # Process each image in the directory
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(images_dir, filename)
            process_image(image_path)
    print("Script execution completed.")
