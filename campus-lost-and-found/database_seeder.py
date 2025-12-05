# File: database_seeder.py
# Purpose: Bootstraps the application with your specific 13-category dataset.
# Operation: Recursively imports images from 'data/raw_dataset', infers category 
#            from folder names, and populates the SQLite database.

import os
import random
import shutil
from modules import db, features, auth

# --- CONFIGURATION ---
SOURCE_FOLDER = "data/raw_dataset"
DEST_FOLDER = "data/uploaded_images"

# Exact match for your 13 dataset folders + "Other"
VALID_CATEGORIES = [
    "Backpack", "Bracelet", "Calculator", "Charger", "Earphones", 
    "Headphones", "Keyboard", "Keys", "Laptop", "Mouse", 
    "Smartphone", "Waterbottle", "Wristwatch", "Other"
]

# Vocabulary for synthetic descriptions
COLORS = ["Blue", "Red", "Black", "White", "Silver", "Green", "Yellow", "Grey"]
LOCATIONS = ["Library", "Gym", "Cafeteria", "Student Center", "Parking Lot", "Room 101", "Main Hall"]

def run_seeding_process():
    print(f"--- [SYSTEM] Starting Database Seeding Protocol ---")
    print(f"--- Source: {SOURCE_FOLDER} ---")
    
    # 0. Initialize Database Tables first
    db.init_db()
    
    # 1. Validation Checks
    if not os.path.exists(SOURCE_FOLDER):
        print(f"[ERROR] Source folder '{SOURCE_FOLDER}' not found.")
        return

    # 2. Authenticate System Admin
    admin_username = "SystemAdmin"
    existing_user = db.get_user_by_username(admin_username)
    
    if existing_user:
        admin_id = existing_user['id']
    else:
        admin_id = auth.register_user(admin_username, "admin_secure_123", "admin@campus.edu")

    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    # 3. Recursive Processing Loop
    success_count = 0
    
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # A. Smart Category Detection
                    folder_name = os.path.basename(root)
                    
                    category = "Other" 
                    for valid_cat in VALID_CATEGORIES:
                        if valid_cat.lower() in folder_name.lower():
                            category = valid_cat
                            break
                    
                    # B. Generate Metadata
                    color = random.choice(COLORS)
                    location = random.choice(LOCATIONS)
                    description = f"Found a {color} {category} near the {location}."
                    
                    # C. File Management
                    src_path = os.path.join(root, filename)
                    safe_filename = f"{folder_name}_{filename}"
                    dest_path = os.path.join(DEST_FOLDER, f"seed_{safe_filename}")
                    
                    shutil.copy(src_path, dest_path)
                    
                    # D. Feature Extraction
                    # FIX: Updated to use the new HOG+Color extractor
                    vis_vector = features.extract_visual_vector(dest_path)
                    text_vector = features.extract_text_vector(description)
                    
                    # E. Database Insertion
                    if vis_vector is not None:
                        db.add_item(
                            user_id=admin_id,
                            item_type="FOUND",
                            category=category,
                            description=description,
                            image_path=dest_path,
                            features_col=vis_vector,
                            features_txt=text_vector
                        )
                        print(f"   [+] Indexed ({category}): {description}")
                        success_count += 1
                        
                except Exception as e:
                    print(f"   [-] Error processing {filename}: {e}")

    print(f"--- [COMPLETED] Database seeded with {success_count} items. ---")

if __name__ == "__main__":
    run_seeding_process()