import cv2
import scipy
from scipy import stats
import numpy as np
from collections import defaultdict

class CartoonEffect:
    def __init__(self):
        self.kernel = np.ones((2, 2), np.uint8)
        self.alpha = 0.001
        self.N = 80
        self.C = [np.array([128])] * 3

    def update_c(self, C, hist):
        while True:
            # Group histogram values based on the closest centroid
            groups = defaultdict(list)
            for i in range(len(hist)):
                if hist[i] == 0:
                    continue
                d = np.abs(C - i)
                index = np.argmin(d)
                groups[index].append(i)

            new_C = np.array(C)
            # Calculate new centroids based on grouped histogram values
            for i, indice in groups.items():
                if np.sum(hist[indice]) == 0:
                    continue
                new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))

            # If centroids have converged, break the loop
            if np.sum(new_C - C) == 0:
                break
            C = new_C

        return C, groups

    def K_histogram(self, hist):
        C = np.array([128])

        while True:
            # Update centroids and group histogram values
            C, groups = self.update_c(C, hist)

            new_C = set()
            # Determine new centroids by analyzing histogram distribution
            for i, indice in groups.items():
                if len(indice) < self.N:
                    new_C.add(C[i])
                    continue

                z, pval = stats.normaltest(hist[indice])
                if pval < self.alpha:
                    left = 0 if i == 0 else C[i - 1]
                    right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                    delta = right - left
                    if delta >= 3:
                        c1 = (C[i] + left) / 2
                        c2 = (C[i] + right) / 2
                        new_C.add(c1)
                        new_C.add(c2)
                    else:
                        new_C.add(C[i])
                else:
                    new_C.add(C[i])

            # If the number of centroids remains unchanged, break the loop
            if len(new_C) == len(C):
                break
            else:
                C = np.array(sorted(new_C))
        return C

    def apply_cartoon_effect(self, img_path):
        # Read the image
        img = cv2.imread(img_path)

        # Apply bilateral filtering to smooth the image while preserving edges
        output = np.array(img)
        x, y, c = output.shape
        for i in range(c):
            output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 150, 150)

        # Detect edges using Canny edge detector
        edge = cv2.Canny(output, 100, 200)

        # Convert image to HSV color space
        output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

        hists = []
        # Compute histograms for each channel (Hue, Saturation, Value)
        hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
        hists.append(hist)
        hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
        hists.append(hist)
        hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
        hists.append(hist)

        # Perform K-Means clustering for each histogram
        for i, h in enumerate(hists):
            self.C[i] = self.K_histogram(h)

        output = output.reshape((-1, c))
        for i in range(c):
            channel = output[:, i]
            # Map pixel values to the closest centroids for each channel
            index = np.argmin(np.abs(channel[:, np.newaxis] - self.C[i]), axis=1)
            output[:, i] = self.C[i][index]

        output = output.reshape((x, y, c))
        output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

        # Draw contours on the output image
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(output, contours, -1, 0, thickness=1)

        # Apply erosion to reduce the thickness of lines
        for i in range(3):
            output[:, :, i] = cv2.erode(output[:, :, i], self.kernel, iterations=1)

        return output

if __name__ == "__main__":
    # Create an instance of the CartoonEffect class
    cartoon_effect = CartoonEffect()

    # Apply the cartoon effect to the input image and save the result
    output_image = cartoon_effect.apply_cartoon_effect("test.png")
    cv2.imwrite("cartoon.jpg", output_image)
