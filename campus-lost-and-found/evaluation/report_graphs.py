# File: evaluation/report_graphs.py
# Purpose: Generates the "Accuracy vs Rank" graph for the final report.
# Simulates 100 "Lost" queries and checks if the correct "Found" item is retrieved.

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simulation Configuration
NUM_ITEMS = 100
CATEGORIES = ["Backpack", "Keys", "Phone", "Wallet", "Laptop", "Bottle"]
COLORS = ["Blue", "Red", "Black", "White", "Silver", "Green"]

def generate_dummy_data():
    """
    Creates synthetic 'Ground Truth' pairs (Lost Item matches Found Item).
    """
    data = []
    for i in range(NUM_ITEMS):
        # Randomly pick attributes
        cat = random.choice(CATEGORIES)
        col = random.choice(COLORS)
        
        # specific_id helps us track if we found the EXACT right item
        item_id = i 
        
        # 'Found' description (The database entry)
        found_desc = f"{col} {cat} found near building {random.randint(1,10)}"
        
        # 'Lost' query (The user search)
        lost_desc = f"I lost my {col} {cat}"
        
        data.append({
            "id": item_id,
            "found_text": found_desc,
            "lost_text": lost_desc
        })
    return data

def run_evaluation():
    print(f"--- Simulating {NUM_ITEMS} Lost & Found Scenarios ---")
    data = generate_dummy_data()
    
    # 1. Train the engine on all "Found" descriptions
    corpus = [d['found_text'] for d in data]
    vectorizer = TfidfVectorizer()
    database_vectors = vectorizer.fit_transform(corpus)
    
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    
    # 2. Run a search for every "Lost" item
    for i, item in enumerate(data):
        query_vec = vectorizer.transform([item['lost_text']])
        
        # Calculate similarity against ALL found items
        scores = cosine_similarity(query_vec, database_vectors).flatten()
        
        # Sort results: Get indices of top matches
        top_indices = scores.argsort()[-10:][::-1]
        
        # Check if the correct ID (i) is in the top K results
        if i in top_indices[:1]: hits_at_k[1] += 1
        if i in top_indices[:3]: hits_at_k[3] += 1
        if i in top_indices[:5]: hits_at_k[5] += 1
        if i in top_indices[:10]: hits_at_k[10] += 1

    # 3. Calculate Percentages
    accuracies = [hits_at_k[k] / NUM_ITEMS * 100 for k in [1, 3, 5, 10]]
    ranks = ["Top-1", "Top-3", "Top-5", "Top-10"]
    
    print(f"Results: {dict(zip(ranks, accuracies))}")
    return ranks, accuracies

def plot_results(ranks, accuracies):
    """
    Draws the graph and saves it as an image.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, accuracies, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    plt.fill_between(ranks, accuracies, color='#3498db', alpha=0.3)
    
    plt.title(f"Retrieval Accuracy (N={NUM_ITEMS} Simulated Items)", fontsize=14)
    plt.xlabel("Rank (k)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 110)
    
    # Add labels on points
    for i, txt in enumerate(accuracies):
        plt.annotate(f"{txt:.1f}%", (ranks[i], accuracies[i]+3), ha='center')
    
    filename = "evaluation_chart.png"
    plt.savefig(filename)
    print(f"[SUCCESS] Chart saved to {filename}")

if __name__ == "__main__":
    try:
        ranks, acc = run_evaluation()
        plot_results(ranks, acc)
    except ImportError:
        print("[ERROR] You need matplotlib installed. Run: pip install matplotlib")