import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Handle paths dynamically for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "Image Classification")

images = []
labels = []

# 2. Match your exact uppercase GitHub folder names
classes = ["CAT", "DOG"]
IMG_SIZE = 64

# 3. Outer loop (no indentation)
for label, folder in enumerate(classes):
    
    # Indented 4 spaces
    folder_path = os.path.join(dataset_path, folder)
    
    # 4. Inner loop to read files (Indented 4 spaces)
    for file in os.listdir(folder_path):
        
        # Everything inside here is indented 8 spaces
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        
        # Fix missing colon and indentation for continue
        if img is None:
            continue
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()
        
        images.append(img)
        labels.append(label)

# 5. Training blocks (completely outside the loops - flush left)
X = np.array(images)
y = np.array(labels)
print("Training Images:", len(X))

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 6. Added quotes around the filename string
joblib.dump(model, "cat_dog_model.pkl")
print("Model Saved Successfully!")
