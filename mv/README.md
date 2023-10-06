## Cartoonify Reality - Image and Video Cartoonization with OpenCV

This project demonstrates how to create a cartoon effect for both images and videos using only core OpenCV filters and functions. The cartoon effect is achieved by applying various image processing techniques, including K-means clustering, bilateral filtering, edge detection (Canny), and erosion.

### Algorithm Used - K-Means Clustering

The core algorithm used in this project is K-Means Clustering. K-Means is an unsupervised machine learning algorithm used for image compression and segmentation. Here, it is applied to group similar color pixels together, giving the image a basic cartoonish appearance.

### Filters Applied

The following OpenCV filters and functions are used to create the cartoon effect:

1. Bilateral Filter: A bilateral filter is applied to smooth the image while preserving edges, resulting in a cartoon-like appearance.

2. Contours: Contours are used to detect edges in the image, which further enhances the cartoon effect.

3. Erosion: Erosion is applied to reduce the thickness of lines, making the cartoon effect more pronounced.

4. Canny Edge Detection: Canny edge detection is used to detect edges, which are then drawn on the image to add the cartoonish tinge.

### Prerequisites

To run the code, you need the following Python libraries:

- scipy
- numpy
- cv2 (OpenCV)

### Getting Started

1. Download a Python interpreter, preferably a version beyond 3.0.

2. Install the prerequisite libraries (scipy, numpy, cv2).

3. Run the `vid.py` file to cartoonize your webcam feed. This will enable real-time cartoonization of your webcam video.