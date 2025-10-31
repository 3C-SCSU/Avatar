import sys

import cv2
import matplotlib.pyplot as plt
import torch
from deepface import DeepFace
from facenet_pytorch import MTCNN

# Initialize MTCNN for face detection
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

# Path to your input image
# image_path = r"C:\Users\Yash Patel\OneDrive\Desktop\image.jpeg"  # Replace with your image pat
image_path = sys.argv[1]  # Get path from command line

# Read and convert the image from BGR to RGB
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found. Check the image path.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces in the image
boxes, probs = mtcnn.detect(img_rgb)

if boxes is not None and len(boxes) > 0:
    # For demonstration, we'll use the first detected face
    x1, y1, x2, y2 = boxes[0].astype(int)
    face_img = img_rgb[y1:y2, x1:x2]

    # Display the detected face
    plt.imshow(face_img)
    plt.title("Detected Face")
    plt.axis("off")
    plt.show()

    # Estimate age using DeepFace on the cropped face image
    # Note: We set enforce_detection=False since we already detected the face.
    analysis = DeepFace.analyze(face_img, actions=["age"], enforce_detection=False)
    if isinstance(analysis, list):
        analysis = analysis[0]
    estimated_age = analysis.get("age", "Age not found")
    print("Estimated Age:", estimated_age)

    # Save the age to a file
    with open("estimated_age.txt", "w") as f:
        f.write(str(estimated_age))
else:
    print("No face detected in the image.")
