# File: train_model.py
import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules import features

# --- CONFIGURATION ---
DATASET_PATH = "data/raw_dataset" 
MODEL_PATH = "modules/category_classifier.pkl"

def load_dataset():
    print("--- [1/3] Loading Dataset & Extracting Features (Color + HOG) ---")
    X = [] 
    y = [] 
    
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        return None, None

    count = 0
    for root, dirs, files in os.walk(DATASET_PATH):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(root, filename)
                folder_name = os.path.basename(root)
                
                try:
                    # Direct call to the new HOG extractor
                    color = features.get_raw_color_hist(filepath)
                    hog_feats = features.get_hog_features(filepath)
                    
                    if color is not None and hog_feats is not None:
                        combined = np.concatenate([color, hog_feats])
                        X.append(combined)
                        y.append(folder_name)
                        count += 1
                        if count % 500 == 0:
                            print(f"   Processed {count} images...")
                        
                except Exception as e:
                    pass
                
    print(f"[INFO] Total Processed: {len(X)} images.")
    return np.array(X), np.array(y)

def train():
    X, y = load_dataset()
    if X is None or len(X) == 0:
        print("[ERROR] No data found.")
        return

    print(f"--- [2/3] Training Random Forest on {len(X)} samples ---")
    # HOG vectors are large (~1700 dims), so we use more trees (n_estimators=150)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"--- [RESULT] Model Accuracy: {acc*100:.2f}% ---")

    print("--- [3/3] Saving Model ---")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()