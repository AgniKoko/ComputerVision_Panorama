# Panorama Creation using Computer Vision

## Project Description
This project involves implementing algorithms to create panoramas from multiple input images using feature detectors and descriptors. The generated panoramas are evaluated using specific metrics to assess their quality and performance.

## Objectives
- **Panorama Creation**: Stitch at least four images using the following feature detectors and descriptors:
  - SIFT ([Documentation](https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html))
  - SURF ([Documentation](https://docs.opencv.org/3.4/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html))
- **Evaluation Metrics**:
  - Differential entropy (𝑓2)
  - Average local entropy (𝑓3)
  - Differential variance of local entropy (𝑓4)
  - Absolute difference of standard deviations (𝑓9)
- **Custom Matching Algorithm**: Implement a cross-checking method for feature matching.

## Features
1. **Feature Detection and Matching**:
   - Implementation of custom cross-checking for bidirectional feature matching.
   - Utilization of OpenCV's SIFT and SURF for feature extraction.
2. **Panorama Stitching**:
   - Pairwise matching of image sets followed by panorama creation.
3. **Quality Evaluation**:
   - Calculation of entropy-based and variance-based metrics to assess panorama quality.

## Data
To execute this project, you need the dataset provided by the course instructor. Download the dataset from [this link](https://vc.ee.duth.gr:6960/index.php/s/QCb6Q5E9lqXjGt4).

## Requirements Used
- Python 3.7.9
- OpenCV 3.4.2.17
- Additional Python libraries:
  - NumPy
  - Matplotlib

## Evaluation Metrics
𝑓2 (Differential Entropy): Measures the difference between the global entropy of the stitched panorama and the average global entropy of the input images.
𝑓3 (Average Local Entropy): Calculates the average local entropy using a 9x9 sliding window.
𝑓4 (Differential Variance of Local Entropy): Measures the difference between the local entropy of the panorama and the input images.
𝑓9 (Absolute Difference of Standard Deviations): Evaluates the consistency of standard deviations between the panorama and the input images.

## Results
Generated panoramas and their corresponding metrics will be saved in the results directory. Example output includes:
- Stitched panorama images
- Metric values and evaluation reports
  
## Notes
This project is based on an assignment from the "Computer Vision" course at Democritus University of Thrace (DUTH). The original task description is intellectual property of the course instructor.
