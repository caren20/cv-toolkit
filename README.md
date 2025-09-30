# Computer Vision Toolkit (OpenCV Project)

This project is a collection of computer vision tasks implemented in Python using **OpenCV** and **NumPy**.  
It provides interactive demonstrations for:

- **Image Processing** (grayscale conversion, thresholding, blurring, manual filters)
- **Edge Detection** (Laplacian, Sobel, and Canny)
- **Shape Detection** (feature matching using ORB and FLANN)
- **Pattern Recognition** (face detection and recognition using Haar cascades and LBPH recognizer)

---

## Features

### 1. Image Processing
- Convert an image to grayscale
- Apply thresholding techniques:
  - Binary, Inverse Binary, Truncation, ToZero, OTSU
- Apply blurring methods:
  - Mean, Gaussian, Median, Bilateral
- Implement manual averaging and median filters

### 2. Edge Detection
- Laplacian operator
- Sobel (X and Y gradients)
- Canny edge detector

### 3. Shape Detection
- Detect keypoints and descriptors using **ORB**
- Perform feature matching with **FLANN-based matcher**
- Visualize matching results

### 4. Pattern Recognition
- Train a face recognizer using **LBPHFaceRecognizer**
- Use Haar Cascade for face detection
- Recognize faces from a test set and display predictions with confidence values

---
