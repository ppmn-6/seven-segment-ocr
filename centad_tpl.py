import os
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Directory containing all images
path_all_images = "/home/wang-zhisheng/Downloads/centad/Images"

# Path to standard image.
path_standard_image = "/home/wang-zhisheng/Downloads/centad/Rotated Images/Pic_2025_05_23_170428_26.bmp"

# Path to templates
path_all_templates = "/home/wang-zhisheng/Downloads/centad/Templates"

# The code below was modified from the following sources:
# https://medium.com/@Hiadore/preprocessing-images-for-ocr-a-step-by-step-guide-to-quality-recovery-923b6b8f926b.
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

# Detect and compute keypoints and descriptors using SIFT.
def detect_and_compute_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Match keypoints between two images using BFMatcher and apply the ratio test.
def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches

# Extract matched points from keypoints based on matches.
def get_matched_points(keypoints1, keypoints2, matches):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2

# Compute the homography matrix using RANSAC.
def compute_homography(points1, points2):
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    return h

# Apply a perspective warp to the image using the homography matrix.
def warp_image(image, h, shape):
    height, width = shape[:2]
    warped_image = cv2.warpPerspective(image, h, (width, height))
    return warped_image

all_images = os.listdir(path_all_images)
standard_image = cv2.imread(path_standard_image, cv2.IMREAD_GRAYSCALE)

for file in all_images:
    #file = "Pic_2025_05_23_170531_104.bmp" # testing
    #path_modified_image = path_all_images + "/" + file
    path_modified_image = "/home/wang-zhisheng/Downloads/centad/standard/Pic_2025_05_23_170428_26.bmp"
    modified_image_original = cv2.imread(path_modified_image)
    modified_image = cv2.imread(path_modified_image, cv2.IMREAD_GRAYSCALE)

    keypoints1, descriptors1 = detect_and_compute_keypoints(standard_image)
    keypoints2, descriptors2 = detect_and_compute_keypoints(modified_image)

    matches = match_keypoints(descriptors1, descriptors2)

    points1, points2 = get_matched_points(keypoints1, keypoints2, matches)

    h = compute_homography(points1, points2)

    # Warp the modified image
    aligned_image = warp_image(modified_image, h, standard_image.shape)

	# Brighten dark regions
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    final_image = clahe.apply(aligned_image)
    
    # Remove noise
    final_image = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    
    # Sharpen image
    kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    final_image = cv2.filter2D(final_image, -1, kernel)
    '''for thresh in range(255):
        _, testing = cv2.threshold(final_image, thresh, 255, cv2.THRESH_BINARY)
        plt.imsave(f"/home/wang-zhisheng/Downloads/centad/testing/{thresh}.png", testing)'''
    # experimental
    #final_image = cv2.adaptiveThreshold(final_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 79, 50)
    
    # Remove small components
    '''num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_image, 4, cv2.CV_32S)
    mask = np.zeros(final_image.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 10:
            mask[labels == i] = 255
    min_area = 0'''
    '''plt.subplot(121),plt.imshow(mask, cmap = 'gray')
    plt.subplot(122),plt.imshow(final_image, cmap = 'gray')
    plt.show()'''
    for digit in range(0, 10):
        all_templates = os.listdir(path_all_templates + f"/{digit}")
        for temp in all_templates:
            template = cv2.imread(path_all_templates + f"/{digit}/{temp}", cv2.IMREAD_GRAYSCALE)
            #template = cv2.adaptiveThreshold(template, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 10)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(final_image, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(aligned_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    plt.subplot(121),plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(aligned_image, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
