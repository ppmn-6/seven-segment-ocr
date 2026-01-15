import os
import argparse
import cv2
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch import nn

parser = argparse.ArgumentParser(description="Seven-Segment OCR")
parser.add_argument("color", help="d if digits are dark, b if digits are bright")
parser.add_argument("images", help="Directory of the batch of images to be analysed")
parser.add_argument("reference_image", help="Path to reference image")
parser.add_argument("reference_ROI", help="Path to cropped ROI of reference image")
parser.add_argument("model", help="Path to model")
parser.add_argument("results", help="File to write results to")

args = parser.parse_args()

color = True

# Check arguments
if args.color != "b" and args.color != "d":
    raise ValueError("Invalid color. Please enter d if digits are dark, b if digits are bright")

if args.color == "d":
    color = False

if not os.path.isdir(args.images):
    raise NotADirectoryError(f"{args.images} is not a directory")

if not os.path.isfile(args.reference_image):
    raise FileNotFoundError(f"{args.reference_image} is not a file")

if not os.path.isfile(args.reference_ROI):
    raise FileNotFoundError(f"{args.reference_ROI} is not a file")

if not os.path.isfile(args.model):
    raise FileNotFoundError(f"{args.model} is not a file")

# All images
path_all_images = args.images
all_images = os.listdir(path_all_images)

# Standard image
path_standard_image = args.reference_image
standard_image = cv2.imread(path_standard_image, cv2.IMREAD_GRAYSCALE)

# Path to cropped ROI of standard image
path_ROI_image = args.reference_ROI
ROI_image = cv2.imread(path_ROI_image, cv2.IMREAD_GRAYSCALE)

# Path to save results
path_results = args.results

# The SIFT code below was modified from
# https://medium.com/@Hiadore/preprocessing-images-for-ocr-a-step-by-step-guide-to-quality-recovery-923b6b8f926b.

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

def process_SIFT(path):
    modified_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = detect_and_compute_keypoints(standard_image)
    keypoints2, descriptors2 = detect_and_compute_keypoints(modified_image)
    matches = match_keypoints(descriptors1, descriptors2)
    points1, points2 = get_matched_points(keypoints1, keypoints2, matches)
    h = compute_homography(points1, points2)
    aligned_image = warp_image(modified_image, h, standard_image.shape)
    return aligned_image

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 14 * 14, out_features=num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Find coordinates of ROI in image
standard_shape = standard_image.shape
ori_height, ori_width = ROI_image.shape
for y in range(standard_shape[0] - ori_height):
    for x in range(standard_shape[1] - ori_width):
        if np.array_equal(standard_image[y:y + ori_height, x:x + ori_width], ROI_image):
            start_y = y
            start_x = x
            end_y = y + ori_height
            end_x = x + ori_width

# Resize ROI image
ROI_image = cv2.resize(ROI_image, (1000, int(ori_height / ori_width * 1000)))
height, width = ROI_image.shape

# Improve contrast
clahe = cv2.createCLAHE(2.0, (8, 8))
ROI_image = clahe.apply(ROI_image)

# Compute skew
upper, _ = cv2.threshold(ROI_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
lower = 0.5 * upper
edges = cv2.Canny(ROI_image, lower, upper)
lines = cv2.HoughLines(edges, 1, np.pi/180, int(height / 10))
angles = []
for line in lines:
    rho, theta = line[0]
    if theta < np.pi/18:
        angles.append(theta)
skew = np.median(np.array(angles)) / np.pi * 180
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, skew, 1.0)
cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])
new_width = int((height * sin) + (width * cos))
new_height = int((height * cos) + (width * sin))
rotation_matrix[0, 2] += (new_width / 2) - center[0]
rotation_matrix[1, 2] += (new_height / 2) - center[1]
rotated_image2 = cv2.warpAffine(ROI_image, rotation_matrix, (new_width, new_height))

model = CNN(1, 7)
model.load_state_dict(torch.load(args.model, weights_only=True)["model_state_dict"])
model.eval()

digits = (0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101,
          0b1111101, 0b0000111, 0b1111111, 0b1101111)

all_results = []

for file in all_images:
    results = []
    total_prob = 1.0

    path_modified_image = path_all_images + "/" + file
    original_image = cv2.imread(path_modified_image)
    final_image = process_SIFT(path_modified_image)

    # Crop image
    final_image = final_image[start_y:end_y, start_x:end_x]

    # Resize image
    final_image = cv2.resize(final_image, (1000, int(ori_height / ori_width * 1000)))

    # save copy for recognition
    rotated_image = cv2.warpAffine(final_image, rotation_matrix, (new_width, new_height))
    if color:
        rotated_image = cv2.bitwise_not(rotated_image)

    # Invert image
    if not color:
        final_image = cv2.bitwise_not(final_image)

    # Even out illumination
    bg = cv2.GaussianBlur(final_image, (51, 51), 0)
    final_image = cv2.subtract(final_image, bg)
    final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)

    # Close up gaps
    kernel = np.ones((5, 5), np.uint8)
    final_image = cv2.dilate(final_image, kernel, iterations = 1)

    # Threshold image
    _, thresh_image = cv2.threshold(final_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Rotate image
    final_image = cv2.warpAffine(final_image, rotation_matrix, (new_width, new_height))
    thresh_image = cv2.warpAffine(thresh_image, rotation_matrix, (new_width, new_height))
    
    # Segment image
    digit = False
    borders = []
    for col in range(new_width):
        count = 0
        for row in range(new_height):
            if thresh_image[row][col]:
                count += 1
        if (digit and count / new_height < 0.02) or (not digit and count / new_height >= 0.02):
            digit = digit ^ 1
            borders.append(col)
    if len(borders) & 1:
        borders.pop()
    for i in range(0, len(borders), 2):
        left = borders[i]
        right = borders[i + 1]
        digit_width = right - left
        if digit_width < 20:
            continue
        digit = False
        vert_borders = []
        for row in range(new_height):
            count = 0
            for col in range(borders[i], borders[i + 1]):
                if thresh_image[row][col]:
                    count += 1
            if (digit and count / digit_width < 0.05) or (not digit and count / digit_width >= 0.05):
                digit = digit ^ 1
                vert_borders.append(row)
        if len(vert_borders) & 1:
            vert_borders.pop()
        j = 0
        while j < len(vert_borders) - 3:
            height1 = vert_borders[j + 1] - vert_borders[j]
            height2 = vert_borders[j + 3] - vert_borders[j + 2]
            gap = vert_borders[j + 2] - vert_borders[j + 1]
            if gap >= new_height // 20:
                # Take whichever side is bigger
                if vert_borders[-1] - vert_borders[j + 2] >= vert_borders[j + 1] - vert_borders[0]:
                    vert_borders = vert_borders[j + 2:]
                else:
                    vert_borders = vert_borders[:j + 2]
                j = 0
            else:
                j += 2
        top = vert_borders[0]
        bottom = vert_borders[-1]
        digit_height = bottom - top
        if digit_height <= new_height // 4:
            continue
        for col in range(left, right):
            count = 0
            for row in range(top, bottom):
                if thresh_image[row][col]:
                    count += 1
            if count / digit_height >= 0.02:
                break
            left += 1
        for col in range(right, left, -1):
            count = 0
            for row in range(top, bottom):
                if thresh_image[row][col]:
                    count += 1
            if count / digit_height >= 0.02:
                break
            right -= 1
        digit_width = right - left
        if digit_width < 20:
            continue
        if digit_width <= 50:
            left = max(0, left - 2 * digit_width)
            digit_width = right - left

        segmented_image = rotated_image[top:bottom, left:right]
        h, w = segmented_image.shape
        mi = 0
        for row in range(h):
            for col in range(w):
                mi = max(mi, segmented_image[row][col])
        mi = int(mi)
        size = max(h, w) * 6 // 5
        t = (size - h) // 2
        b = size - h - t
        l = (size - w) // 2
        r = size - w - l
        segmented_image = cv2.copyMakeBorder(segmented_image, t, b, l, r,
                                borderType=cv2.BORDER_CONSTANT, value=np.percentile(segmented_image, 95))
        segmented_image = cv2.resize(segmented_image, (56, 56))
        segmented_image = torchvision.transforms.functional.to_tensor(segmented_image)
        segmented_image = segmented_image.unsqueeze(0)
        result = -1
        with torch.no_grad():
            preds = model(segmented_image)
            preds = torch.sigmoid(preds)
            best_prob = 0.0
            for digit in range(10):
                mask = digits[digit]
                prob = 1.0
                for seg in range(7):
                    if (1 << seg) & mask:
                        prob *= float(preds[0][seg])
                    else:
                        prob *= 1.0 - float(preds[0][seg])
                best_prob = max(best_prob, prob)
                if prob >= 0.7:
                    result = digit
                    break
            total_prob *= best_prob
            results.append(result)
    all_results.append((total_prob, file, results))

all_results.sort()
all_results.reverse()
with open(args.results, "a") as f:
    for i in range(0, len(all_results)):
        f.write(all_results[i][1] + " " + str(all_results[i][2]))
        f.write("\n")
