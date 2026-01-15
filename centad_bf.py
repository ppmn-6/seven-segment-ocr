import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directory containing all images
path_all_images = "/home/wang-zhisheng/Downloads/centad/Images"

# Path to standard image.
path_standard_image = "/home/wang-zhisheng/Downloads/centad/Rotated Images/Pic_2025_05_23_170428_26.bmp"

# The code below was modified from https://medium.com/@Hiadore/preprocessing-images-for-ocr-a-step-by-step-guide-to-quality-recovery-923b6b8f926b.

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

def identify_digits(coords, coords_shift):
    result = []
    for pos in range(0, len(coords_shift)):
        ma = 0
        mi = 1000
        values = []
        for segment in range(0, 7):
            count = 0
            mean = 0
            test = []
            for x in range(coords[segment][0][0] + coords_shift[pos][0], coords[segment][0][1] + coords_shift[pos][0]):
                for y in range(coords[segment][1][0] + coords_shift[pos][1], coords[segment][1][1] + coords_shift[pos][1]):
                    mean += np.int64(aligned_image[y][x])
                    test.append(np.int64(aligned_image[y][x]))
                    count += 1
            mean /= count
            values.append(mean)
            ma = max(ma, mean)
            mi = min(mi, mean)
            plt.hist(test, bins=255)
            plt.show()
        if ma - mi <= 20:
            result.append(8)
            continue
        on_thres = mi + 0.3 * (ma - mi)
        off_thres = mi + 0.7 * (ma - mi)
        digit = 0
        for segment in range(0, 7):
            if off_thres >= values[segment] and on_thres <= values[segment]:
                return None
            if on_thres > values[segment]:
                digit |= 1 << segment
        isGood = False
        for i in range(0, 10):
            if digit == digits[i]:
                result.append(i)
                isGood = True
                break
        if not isGood:
            return None
    return result

all_images = os.listdir(path_all_images)
standard_image = cv2.imread(path_standard_image, cv2.IMREAD_GRAYSCALE)

# Hardcoded coordinates
big_coords_base = (((147, 168), (213, 223)),
              ((172, 179), (228, 256)),
              ((170, 177), (278, 308)),
              ((144, 164), (312, 322)),
              ((133, 140), (276, 306)),
              ((135, 143), (228, 256)),
              ((144, 168), (264, 269)))
big_coords_shift = ((0, 0), (73, 0), (132, 0), (207, -1), (267, -2))

small_coords_base = (((471, 478), (253, 258)),
                     ((482, 487), (262, 279)),
                     ((480, 485), (293, 312)),
                     ((470, 474), (315, 320)),
                     ((461, 465), (293, 310)),
                     ((463, 466), (262, 279)),
                     ((469, 477), (283, 288)))
small_coords_shift = ((0, 0), (33, 0), (67, 0))

digits = (0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101,
          0b1111101, 0b0000111, 0b1111111, 0b1101111)

for file in all_images:
    print(file)
    path_modified_image = path_all_images + "/" + file
    modified_image_original = cv2.imread(path_modified_image)
    modified_image = cv2.imread(path_modified_image, cv2.IMREAD_GRAYSCALE)

    keypoints1, descriptors1 = detect_and_compute_keypoints(standard_image)
    keypoints2, descriptors2 = detect_and_compute_keypoints(modified_image)

    matches = match_keypoints(descriptors1, descriptors2)

    points1, points2 = get_matched_points(keypoints1, keypoints2, matches)

    h = compute_homography(points1, points2)

    # Warp the modified image
    aligned_image = warp_image(modified_image, h, standard_image.shape)
    result1 = identify_digits(big_coords_base, big_coords_shift)
    result2 = identify_digits(small_coords_base, small_coords_shift)
    if (result1 == None or result2 == None):
        continue
    all_results = result1 + result2
    print(f"{all_results[0]}:{all_results[1]}{all_results[2]}:{all_results[3]}{all_results[4]}.{all_results[5]}{all_results[6]}{all_results[7]}")
    plt.imshow(modified_image_original)
    plt.show()
    break