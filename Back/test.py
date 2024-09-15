# import os
# import cv2
# import numpy as np
# from openvino.inference_engine import IECore, IENetwork

# # Define paths to your folders
# base_path = '/Users/andrew/Documents/Personal/HackMTY/ProjectSauron/Back/Images'
# train_path = os.path.join(base_path, 'train')

# def load_training_images(train_path):
#     images = []
#     image_paths = []
#     for condition_folder in ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']:
#         condition_path = os.path.join(train_path, condition_folder)
#         for filename in os.listdir(condition_path):
#             img_path = os.path.join(condition_path, filename)
#             if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     images.append(img)
#                     image_paths.append(img_path)
#     return images, image_paths

# def preprocess_images(images, size):
#     preprocessed_images = []
#     for img in images:
#         resized_img = cv2.resize(img, (size[1], size[0]))  # cv2.resize uses (width, height)
#         preprocessed_images.append(resized_img)
#     return preprocessed_images

# def find_optimal_size(images):
#     return min([img.shape for img in images], key=lambda x: x[0] * x[1])

# def train_model(preprocessed_images):
#     # Placeholder for model training logic using OpenVINO
#     pass

# def test_model():
#     # Placeholder for model testing logic
#     pass

# def validate_model():
#     # Placeholder for model validation logic
#     pass

# # Main execution
# train_images, train_image_paths = load_training_images(train_path)
# optimal_size = find_optimal_size(train_images)
# preprocessed_train_images = preprocess_images(train_images, optimal_size)


# print(preprocessed_train_images)
# # train_model(preprocessed_train_images)  # Train the model with preprocessed images
# # test_model()  # Test the model
# # validate_model()  # Validate the model



import os
import shutil
import yaml
from PIL import Image
import os
import cv2
import numpy as np
# from openvino.inference_engine import IECore, IENetwork

# Define paths
base_path = '/Users/andrew/Documents/Personal/HackMTY/ProjectSauron/Back'
data_path = os.path.join(base_path, 'Images')
yolov5_path = os.path.join(base_path, 'yolov5')

def setup_yolov5():
    if not os.path.exists(yolov5_path):
        os.system(f"git clone https://github.com/ultralytics/yolov5 {yolov5_path}")
    os.system(f"pip install -r {os.path.join(yolov5_path, 'requirements.txt')}")

def prepare_dataset():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_path, split, 'labels'), exist_ok=True)

    # Move and rename images, create labels
    for condition_folder in os.listdir(os.path.join(data_path, 'train')):
        if condition_folder == '2_polyps':  # We're only interested in polyps for this example
            condition_path = os.path.join(data_path, 'train', condition_folder)
            for img_file in os.listdir(condition_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Randomly assign to train (70%), val (20%), or test (10%)
                    split = np.random.choice(['train', 'val', 'test'], p=[0.7, 0.2, 0.1])
                    
                    # Move and rename image
                    src_path = os.path.join(condition_path, img_file)
                    dst_path = os.path.join(data_path, split, 'images', f"{condition_folder}_{img_file}")
                    shutil.copy(src_path, dst_path)
                    
                    # Create corresponding label file (assuming full image is polyp)
                    img = Image.open(src_path)
                    w, h = img.size
                    label_content = f"0 0.5 0.5 1.0 1.0"  # class x_center y_center width height
                    with open(os.path.join(data_path, split, 'labels', f"{condition_folder}_{img_file.rsplit('.', 1)[0]}.txt"), 'w') as f:
                        f.write(label_content)

def create_data_yaml():
    data_yaml = {
        'path': data_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['polyp']
    }
    
    with open(os.path.join(data_path, 'polyps.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

def train_model():
    os.system(f"python {os.path.join(yolov5_path, 'train.py')} --img 640 --batch 16 --epochs 100 --data {os.path.join(data_path, 'polyps.yaml')} --weights yolov5s.pt")

def validate_model():
    os.system(f"python {os.path.join(yolov5_path, 'val.py')} --weights {os.path.join(yolov5_path, 'runs/train/exp/weights/best.pt')} --data {os.path.join(data_path, 'polyps.yaml')} --task val")

def test_model():
    os.system(f"python {os.path.join(yolov5_path, 'detect.py')} --weights {os.path.join(yolov5_path, 'runs/train/exp/weights/best.pt')} --source {os.path.join(data_path, 'test/images')} --conf 0.25")

if __name__ == "__main__":
    setup_yolov5()
    prepare_dataset()
    create_data_yaml()
    train_model()
    validate_model()
    test_model()