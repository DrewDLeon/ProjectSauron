import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# Paths
base_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron', 'Back')
valid_path = os.path.join('C:', os.sep, 'Users', 'Andres', 'Documents', 'HackMTY', 'ProjectSauron')
existing_data_yaml = os.path.join(base_path, 'yolov8', 'config.yaml')

# Initialize YOLO model
model_path = os.path.join(valid_path, 'runs', 'detect', 'train10', 'weights', 'best.pt')
model = YOLO(model_path)

# Function to select an image file manually
def seleccionar_imagen():
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", ".png;.jpg;.jpeg;.bmp;*.gif")]
    )
    return file_path

# Function to perform YOLO detection
def analyze_image_yolo(image):
    results = model(image)
    
    # Process the detection results and extract bounding boxes
    for result in results:
        boxes = result.boxes.xyxy  # Get the bounding boxes (x1, y1, x2, y2)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Convert box coordinates to integers
            # Draw a blue rectangle (BGR color format)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue rectangle
    return image


# Function to import and analyze an image
def importar_y_analizar():
    image_path = seleccionar_imagen()
    if not image_path:
        print("No se seleccionó ninguna imagen.")
        return
    
    # Load the image and perform YOLO detection
    image = cv2.imread(image_path)
    annotated_image = analyze_image_yolo(image)
    
    # Update the displayed image with YOLO detections
    img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    label_webcam.imgtk = imgtk
    label_webcam.configure(image=imgtk)

# Define the window
window = tk.Tk()
window.title("Proyecto Sauron")
#window.iconbitmap("img01.ico")

# Define the frames
frame_left = tk.Frame(window, width=300, height=400, bg="#333333")
frame_left.pack(side="left", fill="both", expand=False)

frame_right = tk.Frame(window, width=600, height=400)
frame_right.pack(side="right", fill="both", expand=True)

# Define the label for patient name
label_patient = tk.Label(frame_left, text="Paciente:", bg="#333333", fg="white")
label_patient.pack(pady=10)

# Define the entry for patient name
entry_patient = tk.Entry(frame_left, width=25)
entry_patient.pack()

# Define the label for anomaly detected
label_anomaly = tk.Label(frame_left, text="Anomalía detectada:", bg="#333333", fg="white")
label_anomaly.pack(pady=10)

# Define the entry for anomaly detected
entry_anomaly = tk.Entry(frame_left, width=25)
entry_anomaly.pack()

# Define the button to start the diagnosis
def start_diagnosis():
    ret, frame = cap.read()
    if ret:
        # Perform YOLO detection on the live camera feed
        annotated_frame = analyze_image_yolo(frame)
        
        # Convert the frame for Tkinter display
        img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label_webcam.imgtk = imgtk
        label_webcam.configure(image=imgtk)

        # Call the function again after 10 milliseconds to process the next frame
        label_webcam.after(10, start_diagnosis)
    else:
        print("Error: Cannot capture image.")

button_start = tk.Button(frame_left, text="Iniciar diagnóstico", command=start_diagnosis, bg="#4CAF50", fg="white")
button_start.pack(pady=10)

# Define the button to import and analyze the image
button_import = tk.Button(frame_left, text="Importar y analizar imagen", command=importar_y_analizar, bg="#2196F3", fg="white")
button_import.pack(pady=10)

# Define the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    window.destroy()

# Define the function to capture the image
def capture():
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured!")
    else:
        print("Error: Cannot capture image.")

# Define the button to capture the image
button_capture = tk.Button(frame_right, text="CAPTURAR", command=capture, bg="#2196F3", fg="white", width=15)
button_capture.pack(pady=20)

# Define the function to finalize the diagnosis and close the program
def finalize():
    window.destroy()

# Define the button to finalize the diagnosis and close the program
button_finalize = tk.Button(frame_right, text="FINALIZAR", command=finalize, bg="#2196F3", fg="white", width=15)
button_finalize.pack(pady=20)

# Define the function to show the webcam feed
def show_webcam():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_webcam.imgtk = imgtk
        label_webcam.configure(image=imgtk)
    label_webcam.after(10, show_webcam)  # Update every 10 ms

# Define the label for the webcam feed
label_webcam = tk.Label(frame_right)
label_webcam.pack()

# Start the webcam feed
show_webcam()

# Set the closing protocol
def on_closing():
    cap.release()  # Release the webcam
    window.destroy()  # Close the Tkinter window

window.protocol("WM_DELETE_WINDOW", on_closing)

# Run the main loop
window.mainloop()