# File: modules/auth.py
# Purpose: Manages user security and authentication logic.
# Security: Uses SHA-256 hashing so plain passwords are never stored.

import hashlib
from modules import db

def hash_password(password):
    """
    Converts a plain text password into a SHA-256 hash.
    Example: 'secret' -> '2bb80d53...'
    """
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password):
    """
    Verifies credentials against the database.
    Returns: User object (dict) if valid, None if invalid.
    """
    # 1. Fetch user from DB
    user = db.get_user_by_username(username)
    
    if not user:
        return None  # User does not exist

    # 2. Hash the provided password
    input_hash = hash_password(password)

    # 3. Compare with stored hash
    # Note: We access columns by name because we set row_factory in db.py
    if input_hash == user['password_hash']:
        return user
    
    return None  # Wrong password

def register_user(username, password, contact_info):
    """
    Creates a new user with a hashed password.
    Returns: New User ID or None if username exists.
    """
    # Hash the password before sending to DB
    secure_hash = hash_password(password)
    
    return db.add_user(username, secure_hash, contact_info)