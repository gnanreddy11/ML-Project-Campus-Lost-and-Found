# Campus Lost and Found with AutoMatch

# Done By:-

# Name: P Harshini Sai | Roll no: SE25MAID003

# Name: Pranav P | Roll no: SE25MAID034

# Name: K V VINAY KUMAR | Roll no: SE25MAID049

# Name: Gnan Reddy | Roll no: SE25MAID051

## 1. Project Overview

**Campus Lost and Found with AutoMatch** is a machine-learning powered web application designed to automate the matching of lost and found items.

The system utilizes a **Hybrid Matching Engine** that ranks matches based on:

- **Visual Similarity:** HSV Color Histograms (OpenCV) & HOG Features.
- **Textual Similarity:** TF-IDF Vectors (Scikit-Learn).

**Note:** This project strictly adheres to the constraint of using **Classical ML techniques** (Random Forest, HOG) and avoids Deep Learning/CNNs.

---

## 2. Key Results (Summary)

- **Auto-Classification Accuracy:** ~62% (Random Forest Model).
  - _Context:_ The model classifies uploaded images into 13 categories (e.g., Keys, Laptop) significantly better than random guessing (7.6%).
- **Search Retrieval Accuracy (Top-5):** 98.0%.
  - _Context:_ In a simulation of 100 queries, the correct "Found" item appeared on the first page of results 98% of the time.
- **Interpretability:** The system provides "Matched on" tags (e.g., `blue`, `backpack`) to explain algorithmic decisions to the user.

---

## 3. Setup Instructions

### Prerequisites

- **Python 3.12+**
- **VS Code** (Recommended)

### Installation

1.  Clone/Download this repository.
2.  Open a terminal in the root folder.
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. How to Run the App

### A. Initialize the Database

The project includes a representative subset of the dataset in `data/raw_dataset/`. You must run the seeder script to index these images before searching.

1.  Run the seeder:
    ```bash
    python database_seeder.py
    ```
    _Output:_ You will see `[+] Indexed...` messages for ~130 items.

### B. Launch the App

Start the web interface:

````bash
streamlit run app.py

The app will open at http://localhost:8501.

### C. Default Login
* **Username:** `SystemAdmin`
* **Password:** `admin123`
* *(Alternatively, register a new account on the Login page)*

---

## 5. Technical Architecture & Constraints

### ⚠️ Note on Dataset
The Random Forest model (`modules/category_classifier.pkl`) was trained locally on the full **6,493 image dataset**. To keep this submission file size manageable, the `data/raw_dataset/` folder contains a **curated subset (~130 images)** sufficient for demonstration.

### Technical Stack
* **Classification:** Random Forest (Scikit-Learn).
* **Vision:** Histogram of Oriented Gradients (HOG) + HSV Color.
* **NLP:** TF-IDF Vectorization + Cosine Similarity.

---

## 6. Reproducing Evaluation Metrics
To generate the accuracy graph used in the report:

```bash
python evaluation/report_graphs.py
````

- **Output:** Generates `evaluation_chart.png` demonstrating retrieval performance.
