
# Clustering Algorithms for Hyperspectral Image Analysis

This repository contains a comprehensive study, focusing on the analysis and unsupervised classification (clustering) of hyperspectral images (HSIs). The project uses the Salinas HSI dataset, representing an area in the Salinas Valley, California, USA. This dataset consists of a 220x120 spatial resolution image with 204 spectral bands, featuring 26,400 pixels categorized into seven ground-truth classes. The project explores different clustering techniques to identify homogeneous regions within the hyperspectral image. Both **cost function optimization algorithms** (k-means, fuzzy c-means, possibilistic c-means, probabilistic c-means) and **hierarchical clustering methods** (Complete-link, WPGMC, Ward’s algorithm) are applied and evaluated.

This analysis was performed as part of the final project for the "Clustering Algorithms" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Konstantinos Koutroumbas, in the academic year 2023-2024.

---

## Project Overview

### Objectives
- Perform clustering on the Salinas HSI dataset to detect homogeneous regions.
- Compare the effectiveness of cost function optimization and hierarchical clustering algorithms.
- Evaluate clustering performance both qualitatively (based on ground-truth labels and principal component analysis) and quantitatively (using cluster validation metrics like silhouette scores).

### Workflow
1. **Data Preprocessing**: 
   - Conversion of the HSI cube into a 2D dataset suitable for clustering.
   - Mean normalization and Principal Component Analysis (PCA) to reduce dimensionality while preserving variance.

2. **Clustering Execution**:
   - Apply algorithms with various parameter configurations.
   - Analyze results for clusters ranging from 4 to 7.

3. **Performance Evaluation**:
   - Qualitative: Compare cluster assignments with ground-truth labels and principal components.
   - Quantitative: Use silhouette scores and validation metrics to assess clustering accuracy.

4. **Visualization**:
   - Generate plots for clustering results versus ground-truth labels in PCA-reduced space.

---

## Results Summary

The project demonstrated varying levels of performance across algorithms:
- **Cost Function Optimization Algorithms**: 
  - K-means and fuzzy c-means showed robust results but were sensitive to initialization.
  - Possibilistic c-means faced challenges in producing the exact desired cluster counts.
  - Probabilistic c-means provided probabilistic insights but required more computational resources.

- **Hierarchical Clustering**:
  - Algorithms such as Complete-link and Ward’s performed well, especially in identifying non-compact cluster shapes.
  
Principal Component Analysis effectively reduced data dimensions while preserving cluster separability, aiding in visual and computational analysis.

---

## Cloning the Repository
To get a copy of this project, run:
```bash
git clone https://github.com/GiatrasKon/Hyperspectral-Image-Clustering.git
```

## Dependencies

This project uses MATLAB. Ensure the following toolboxes are installed:

- Statistics and Machine Learning Toolbox
- Image Processing Toolbox

## Running the Code

1. Open `HSI_clustering_analysis.m` in MATLAB.
2. Ensure that the provided files `Salinas_cube.mat` and `Salinas_gt.mat` are in the `data/` directory before running the script.
3. Configure the `DISPLAY_FIGURES` flag to visualize results.
4. Run the script to process the dataset and generate output.

## Documentation

Refer to the `documents` directory for the project description and report.

---