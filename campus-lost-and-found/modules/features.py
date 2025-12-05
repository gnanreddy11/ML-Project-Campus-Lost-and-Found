# File: modules/features.py
# Purpose: Advanced Feature Extraction (Color + HOG) & Auto-Classification.
# Tech: OpenCV (HSV Histograms), Scikit-Image (HOG), Scikit-Learn (TF-IDF).

import cv2
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
from skimage.transform import resize

# --- CONFIGURATION ---
MODEL_PATH = "modules/category_classifier.pkl"
SEED_VOCAB = [
    "blue", "red", "green", "black", "white", "silver", "gold", "yellow", "grey", "orange", "purple",
    "keys", "wallet", "phone", "iphone", "samsung", "laptop", "macbook", "dell",
    "backpack", "bag", "purse", "bottle", "water", "hydroflask",
    "jacket", "coat", "hoodie", "sweater", "glasses", "sunglasses",
    "id", "card", "student", "license", "credit",
    "bracelet", "calculator", "charger", "earphones", "headphones", 
    "keyboard", "mouse", "smartphone", "waterbottle", "wristwatch",
    "lost", "found", "campus", "library", "cafeteria", "gym", "room", "hall"
]

text_engine = TfidfVectorizer(vocabulary=SEED_VOCAB)
text_engine.fit(SEED_VOCAB)

_classifier = None

def load_ml_model():
    global _classifier
    if _classifier is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _classifier = pickle.load(f)
    return _classifier

# ==========================
# 1. VISUAL FEATURES (COLOR + HOG)
# ==========================

def get_raw_color_hist(image_path):
    """
    Extracts Color Histogram (The 'Paint' of the object).
    """
    img = cv2.imread(image_path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 8x8 bins = 64 features. Compact and fast.
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def get_hog_features(image_path):
    """
    FEATURE ENGINEERING UPGRADE: HOG (Histogram of Oriented Gradients).
    Captures texture and edge directions (e.g., differentiates keys from a phone).
    """
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. Resize to fixed standard size (Critical for HOG)
    # We use 64x64 pixels to keep the vector size manageable (~1000 features)
    resized_img = resize(img, (64, 64))
    
    # 2. Calculate HOG
    # orientations=9, pixels_per_cell=(8, 8) -> Standard config
    # channel_axis=-1 handles RGB images automatically
    hog_features = hog(
        resized_img, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=False, 
        channel_axis=-1
    )
    
    return hog_features

def extract_visual_vector(image_path):
    """
    Combines Color (64 feats) + HOG (~1700 feats) into one robust vector.
    """
    color = get_raw_color_hist(image_path)
    hog_feats = get_hog_features(image_path)
    
    if color is None or hog_feats is None: return None
    
    # Concatenate features
    # Color is small, so we might want to weigh it, but Random Forest handles scale well.
    combined = np.concatenate([color, hog_feats])
    return pickle.dumps(combined)

def predict_category(image_path):
    """
    Predicts category using the improved Color+HOG vector.
    """
    clf = load_ml_model()
    if clf is None: return None
    
    try:
        color = get_raw_color_hist(image_path)
        hog_feats = get_hog_features(image_path)
        
        if color is not None and hog_feats is not None:
            features = np.concatenate([color, hog_feats])
            prediction = clf.predict([features])
            return prediction[0]
    except:
        return None
    return None

def get_visual_similarity(blob_a, blob_b):
    """
    Compares two vectors using Cosine Similarity.
    """
    if blob_a is None or blob_b is None: return 0.0
    vec_a = pickle.loads(blob_a)
    vec_b = pickle.loads(blob_b)
    score = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]
    return float(score)

# ==========================
# 2. TEXT FEATURES & XAI (Unchanged)
# ==========================
def extract_text_vector(text):
    if not text: text = ""
    vector = text_engine.transform([text.lower()]).toarray()
    return pickle.dumps(vector)

def get_text_similarity(blob_a, blob_b):
    if blob_a is None or blob_b is None: return 0.0
    vec_a = pickle.loads(blob_a)
    vec_b = pickle.loads(blob_b)
    score = cosine_similarity(vec_a, vec_b)[0][0]
    return float(score)

def explain_text_match(text_query, text_candidate):
    if not text_query or not text_candidate: return []
    vocab_words = text_engine.get_feature_names_out()
    try:
        vec_query = text_engine.transform([text_query.lower()]).toarray()[0]
        vec_candidate = text_engine.transform([text_candidate.lower()]).toarray()[0]
        overlap_indices = np.where((vec_query > 0) & (vec_candidate > 0))[0]
        return [vocab_words[i] for i in overlap_indices]
    except: return []

# ==========================
# 3. HYBRID MATCHING
# ==========================
def calculate_hybrid_score(vis_blob_a, vis_blob_b, text_blob_a, text_blob_b):
    score_vis = get_visual_similarity(vis_blob_a, vis_blob_b)
    score_text = get_text_similarity(text_blob_a, text_blob_b)
    # HOG is very accurate, so we trust Visuals more (60%)
    return (0.6 * score_vis) + (0.4 * score_text)