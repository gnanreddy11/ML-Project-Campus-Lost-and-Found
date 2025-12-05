# File: app.py
# Purpose: Main entry point. Connects UI (Streamlit) to Backend (Auth, DB, ML).
# Final Version: Professional UI, No Balloons, Fixed DB Access.

import streamlit as st
import os
import time
from modules import auth, db, features

# --- CONFIGURATION ---
UPLOAD_FOLDER = "data/uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.set_page_config(page_title="Campus Lost & Found", layout="wide")

# --- SESSION STATE MANAGEMENT ---
if 'user' not in st.session_state:
    st.session_state['user'] = None

# ==========================
# HELPER FUNCTIONS
# ==========================
def save_uploaded_file(uploaded_file):
    """Saves uploaded image to disk and returns the path."""
    try:
        timestamp = int(time.time())
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def login_page():
    st.title("🔐 Login")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            user = auth.login_user(username, password)
            if user:
                st.session_state['user'] = user
                st.success(f"Welcome back, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        contact = st.text_input("Contact Email/Phone")
        if st.button("Create Account"):
            if new_user and new_pass and contact:
                user_id = auth.register_user(new_user, new_pass, contact)
                if user_id:
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                st.warning("Please fill in all fields.")

def dashboard_page():
    user = st.session_state['user']
    st.sidebar.title(f"👤 {user['username']}")
    if st.sidebar.button("Log Out"):
        st.session_state['user'] = None
        st.rerun()

    menu = st.sidebar.radio("Navigation", ["Report Item", "Search Matches"])

    if menu == "Report Item":
        report_item_page(user)
    elif menu == "Search Matches":
        match_page(user)

def report_item_page(user):
    st.header("📝 Report a Lost or Found Item")
    
    col1, col2 = st.columns(2)
    
    # Initialize session state for auto-detected category
    if 'detected_category' not in st.session_state:
        st.session_state['detected_category'] = 0 
        
    with col2:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", width=200)
            
            # Auto-Classification Logic
            temp_path = save_uploaded_file(uploaded_file)
            prediction = features.predict_category(temp_path)
            
            if prediction:
                st.info(f"🤖 AI detected: **{prediction}**")
                
                options = [
                    "Backpack", "Bracelet", "Calculator", "Charger", "Earphones", 
                    "Headphones", "Keyboard", "Keys", "Laptop", "Mouse", 
                    "Smartphone", "Waterbottle", "Wristwatch", "Other"
                ]
                
                for idx, opt in enumerate(options):
                    if prediction.lower() == opt.lower():
                        st.session_state['detected_category'] = idx
                        break

    with col1:
        item_type = st.selectbox("I have...", ["LOST something", "FOUND something"])
        db_type = "LOST" if "LOST" in item_type else "FOUND"
        
        category = st.selectbox("Category", [
            "Backpack", "Bracelet", "Calculator", "Charger", "Earphones", 
            "Headphones", "Keyboard", "Keys", "Laptop", "Mouse", 
            "Smartphone", "Waterbottle", "Wristwatch", "Other"
        ], index=st.session_state['detected_category'])
        
        description = st.text_area("Description (e.g., 'Blue Nike backpack')")

    if st.button("Submit Report"):
        if not description or not uploaded_file:
            st.error("Please provide both a description and an image.")
            return

        with st.spinner("Processing advanced features..."):
            image_path = save_uploaded_file(uploaded_file)
            
            # Extract Advanced Features
            vis_vector = features.extract_visual_vector(image_path)
            text_vector = features.extract_text_vector(description)
            
            if image_path and vis_vector is not None:
                db.add_item(user['id'], db_type, category, description, image_path, vis_vector, text_vector)
                st.success("Item reported successfully! Algorithm is now indexing...")
            else:
                st.error("Failed to process image.")

def match_page(user):
    st.header("🔍 Smart Match System")
    
    if 'search_active' not in st.session_state:
        st.session_state['search_active'] = False

    search_type = st.radio("Show me...", ["Potential matches for my LOST items", "Potential owners for items I FOUND"])
    target_type = "FOUND" if "LOST" in search_type else "LOST"
    
    candidates = db.get_candidates(target_type)
    
    if not candidates:
        st.info(f"No active '{target_type}' items found in the database yet.")
        return

    st.subheader("Start Your Search")
    
    col1, col2 = st.columns(2)
    with col1:
        query_text = st.text_input("Describe the item:")
    with col2:
        query_image = st.file_uploader("Upload an image (Optional)", type=["jpg", "png", "jpeg"])

    if st.button("Find Matches"):
        st.session_state['search_active'] = True

    if st.session_state['search_active']:
        results = []
        
        query_text_vec = features.extract_text_vector(query_text) if query_text else None
        
        query_vis_vec = None
        if query_image:
             temp_path = save_uploaded_file(query_image)
             query_vis_vec = features.extract_visual_vector(temp_path)

        for item in candidates:
            score = 0.0
            
            if query_text_vec is not None and query_vis_vec is not None:
                score = features.calculate_hybrid_score(
                    query_vis_vec, item['features_color'], 
                    query_text_vec, item['features_text']
                )
            elif query_text_vec is not None:
                score = features.get_text_similarity(query_text_vec, item['features_text'])
            elif query_vis_vec is not None:
                score = features.get_visual_similarity(query_vis_vec, item['features_color'])
            
            if score > 0.0:
                results.append((score, item))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        if results:
            st.write(f"Found {len(results)} matches:")
            for score, item in results:
                color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                
                matched_keywords = []
                if query_text:
                    matched_keywords = features.explain_text_match(query_text, item['description'])
                
                with st.container():
                    st.divider()
                    c1, c2 = st.columns([1, 3])
                    
                    with c1:
                        if item['image_path'] and os.path.exists(item['image_path']):
                            st.image(item['image_path'], use_container_width=True)
                        else:
                            st.caption("No Image")
                    
                    with c2:
                        st.subheader(f"{item['category']} ({int(score*100)}% Match)")
                        st.markdown(f"**Description:** {item['description']}")
                        
                        if matched_keywords:
                            tags = " ".join([f"`{word}`" for word in matched_keywords])
                            st.markdown(f"**Matched on:** {tags}")
                        
                        st.markdown(f"**Confidence:** :{color}[{score:.2f}]")
                        
                        # FIXED: Use dictionary access for contact info and removed balloons
                        if st.button("✅ Confirm Match", key=f"claim_{item['id']}"):
                            st.success(f"Item confirmed! Owner contact revealed: {item['contact_info']}")
        else:
            st.warning("No matches found. Try adjusting your description.")

def main():
    db.init_db()
    if st.session_state['user']:
        dashboard_page()
    else:
        login_page()

if __name__ == "__main__":
    main()