# Fashion Tagger AI: Image Retrieval & Classification

A Computer Vision project that automatically tags clothing images by **Color** (using Unsupervised Learning) and **Clothing Type** (using Supervised Learning).

This project implements **K-Means** and **K-Nearest Neighbors (K-NN)** algorithms entirely from scratch using NumPy, focusing on mathematical optimization and heuristic analysis.

## Project Overview

The goal of this system is to label fashion images to enable an intelligent retrieval system (e.g., "Find me all *Blue* *Flip Flops*").

* **Color Detection:** Uses **K-Means Clustering** to identify dominant colors in an image.
* **Shape Classification:** Uses **K-NN** to categorize the item (e.g., Sandals, Jeans, Shirts).
* **Optimization:** The project explores different initialization methods (Random, Custom) and fitting metrics (Fisher Discriminant, Inter-Class Distance) to improve accuracy and speed.

## Technologies & Algorithms

* **Python 3.x**
* **NumPy & SciPy:** For matrix operations and distance calculations.
* **Matplotlib:** For visualizing clusters and results.
* **No ML Libraries:** The core logic of K-Means and K-NN is built without Scikit-Learn to demonstrate algorithmic understanding.

## Repository Structure

* `my_labeling.py`: The main script. Handles data loading, executes the retrieval queries, and runs performance statistics.
* `Kmeans.py`: Custom implementation of the K-Means algorithm.
    * *Features:* Supports 'random', 'first', and 'custom' centroid initialization. Implements WCD (Within-Class Distance) and Fisher Discriminant for optimal 'K' selection.
* `KNN.py`: Custom implementation of K-Nearest Neighbors.
    * *Features:* Uses grayscale conversion to speed up shape classification by ~8x while maintaining accuracy.
* `utils.py` & `utils_data.py`: Helper functions for image processing and dataset management.

## Algorithmic Details

### 1. K-Means (Color Labeling)
The system treats image pixels as points in a 3D space (RGB).
* **Best K Selection:** Implemented heuristics to automatically find the optimal number of colors (K) using the **Fisher Discriminant**, which maximizes Inter-Class Distance (ICD) and minimizes Within-Class Distance (WCD).
* **Initialization:** Compared standard initialization vs. equidistant selection to reduce convergence time.

### 2. K-NN (Shape Classification)
Classifies the type of clothing based on visual similarity.
* **Grayscale Optimization:** The project demonstrates that converting images to grayscale before processing significantly reduces computational load (dimensionality reduction) without sacrificing classification accuracy (~91% accuracy maintained).

## Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/Fashion-Tagger-AI.git](https://github.com/yourusername/Fashion-Tagger-AI.git)
    ```

2.  **Run the Labeler:**
    You can run the main script to see retrieval examples (e.g., searching for "Blue" items):
    ```bash
    python my_labeling.py
    ```

3.  **Run Statistical Analysis:**
    Uncomment the function calls in `my_labeling.py` (like `Kmean_statistics` or `millora_knn`) to generate the performance graphs shown in the report.

## Key Results

* **Accuracy:** Achieved ~91% accuracy in shape classification using K-NN with `k=4`.
* **Performance:** Grayscale conversion reduced K-NN execution time drastically compared to RGB processing.
* **Color Segmentation:** Fisher Discriminant proved to be the most robust metric for separating distinct color clusters.

## Authors

* Nerea de la Torre Veguillas
* Mara Montero Jurado
* Júlia Morán Fluvià
* Adrián Prego Gallart

Computational Mathematics & Data analyitics, UAB
