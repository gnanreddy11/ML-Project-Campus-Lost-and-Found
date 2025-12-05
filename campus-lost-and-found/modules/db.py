# File: modules/db.py
import sqlite3
import os

DB_FOLDER = "data"
DB_NAME = "campus.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

def get_connection():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        contact_info TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        status TEXT DEFAULT 'OPEN',
        type TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        image_path TEXT,
        features_color BLOB,
        features_text BLOB,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """)
    conn.commit()
    conn.close()

def add_user(username, password_hash, contact_info):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, contact_info) VALUES (?, ?, ?)", 
                       (username, password_hash, contact_info))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_username(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def add_item(user_id, item_type, category, description, image_path, features_col, features_txt):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO items (user_id, status, type, category, description, image_path, features_color, features_text)
        VALUES (?, 'OPEN', ?, ?, ?, ?, ?, ?)
    """, (user_id, item_type, category, description, image_path, features_col, features_txt))
    conn.commit()
    item_id = cursor.lastrowid
    conn.close()
    return item_id

def get_candidates(target_type):
    """
    Retrieves matches AND joins with users table to get contact info.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # CRITICAL FIX: explicit JOIN to fetch contact_info
    cursor.execute("""
        SELECT items.id, items.category, items.description, items.image_path, 
               items.features_color, items.features_text, users.contact_info
        FROM items 
        JOIN users ON items.user_id = users.id 
        WHERE items.type = ? AND items.status = 'OPEN'
    """, (target_type,))
    candidates = cursor.fetchall()
    conn.close()
    return candidates