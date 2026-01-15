import os
import cv2
import math
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''dir = "/home/wang-zhisheng/Downloads/centad/dataset/testing9/"
out_dir = "/home/wang-zhisheng/Downloads/centad/dataset/testing10/"
files = [f for f in os.listdir(dir) if os.path.isfile(dir + f)]

def scaling(size, out):
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    result = cv2.copyMakeBorder(out, top, bottom, left, right,
                            borderType=cv2.BORDER_CONSTANT, value=mi)
    result = cv2.resize(result, (56, 56))
    return result

# do this last
def rotate():
    M = cv2.getRotationMatrix2D((w // 2, h // 2), random.randint(-10, 10), 0.8)
    output = cv2.warpAffine(img, M, (w, h))
    output[output < mi2 + 30] = mi
    if random.randint(0, 1) == 1:
        x = max(h, w) * 6 // 5
    else:
        x = max(h, w) * 1000 // 999
    return scaling(x, output)

# angle: [-pi, pi], strength: [10, 50]
def grad(angle, strength):
    dx = np.cos(angle)
    dy = np.sin(angle)
    x = np.linspace(-0.5, 0.5, w)
    y = np.linspace(-0.5, 0.5, h)
    xv, yv = np.meshgrid(x, y)
    gradient = dx * xv + dy * yv
    gradient = strength * gradient
    output = img.astype(np.float32) + gradient.astype(np.float32)
    output = np.clip(output, 0, 255)
    return output

for file in files:
    s1 = file[:-6]
    l = s1.split("_")
    mask = int(l[-1])
    label = []
    for seg in range(7):
        if mask % 3 == 1:
            label.append(0.5)
        elif mask % 3 == 2:
            label.append(1)
        else:
            label.append(0)
        mask //= 3
    with open("/home/wang-zhisheng/Downloads/centad/dataset/testing7.csv", "a") as f:
        for i in range(5):
            f.write(f'"{file}_{i + 2}.png"')
            for x in label:
                f.write("," + str(x))
            f.write("\n")
    
    img = cv2.imread(dir + file, cv2.IMREAD_GRAYSCALE)
    ori_img = img.copy()
    h, w = img.shape

    mi2 = 255
    for row in range(h):
        for col in range(w):
            mi2 = min(mi2, img[row][col])

    for i in range(5 * 3):
        img = ori_img.copy()
        mi = np.percentile(img, random.randint(10, 20))
        img = grad(random.uniform(-math.pi, math.pi), random.randint(0, 20))
        img = rotate()
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.bitwise_not(img)
        cv2.imwrite(f"/home/wang-zhisheng/Downloads/centad/dataset/testing10/{file}_{i + 5}.png", img)'''

'''with open("/home/wang-zhisheng/Downloads/centad/dataset/training_new.csv", "r") as file, open("/home/wang-zhisheng/Downloads/centad/dataset/training.csv", "a") as file2:
    for line in file:
        all = line.split(",")
        pic = all[0][1:-1]
        if "new" in pic and "flip" not in pic:
            os.remove(f"/home/wang-zhisheng/Downloads/centad/dataset/training/{pic}")
        else:
            file2.write(line)

        if random.randint(0, 99) >= 80:
            with open("/home/wang-zhisheng/Downloads/centad/dataset/testing.csv", "a") as file2:
                file2.write(line)
            shutil.move(f"/home/wang-zhisheng/Downloads/centad/dataset/testing7/{pic}",
                        f"/home/wang-zhisheng/Downloads/centad/dataset/testing/{pic}")
        else:
            with open("/home/wang-zhisheng/Downloads/centad/dataset/training.csv", "a") as file2:
                file2.write(line)
            shutil.move(f"/home/wang-zhisheng/Downloads/centad/dataset/testing7/{pic}",
                        f"/home/wang-zhisheng/Downloads/centad/dataset/training/{pic}")'''

'''dir = "/home/wang-zhisheng/Downloads/centad/dataset/training/"
labels = "/home/wang-zhisheng/Downloads/centad/dataset/training_new.csv"
with open(labels, "a") as f:
    for file in os.listdir(dir):
        f.write(f'"{file}"')
        if "S" in file:
            s1 = file[:-12]
            l = s1.split("_")
            mask = int(l[-1])
            for seg in range(7):
                if mask % 3 == 1:
                    f.write(",0.5")
                elif mask % 3 == 2:
                    f.write(",1")
                else:
                    f.write(",0")
                mask //= 3
        else:
            digit = int(file[0])
            digits = (0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101,
                            0b1111101, 0b0000111, 0b1111111, 0b1101111)
            mask = digits[digit]
            for seg in range(7):
                f.write(f",{mask & 1}")
                mask >>= 1
        f.write("\n")'''

'''with open("/home/wang-zhisheng/Downloads/centad/dataset/training.csv", "r") as file:
    print(sum(1 for line in file))'''

'''with open("/home/wang-zhisheng/Downloads/centad/dataset/testing8.csv", "a") as l:
    for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/training/"):
        if "new" in file:
            img = cv2.imread(f"/home/wang-zhisheng/Downloads/centad/dataset/training/{file}", cv2.IMREAD_GRAYSCALE)
            img = cv2.bitwise_not(img)
            cv2.imwrite(f"/home/wang-zhisheng/Downloads/centad/dataset/testing8/{file}_flip.png", img)

            mask = int(file.split("_")[-4])
            
            l.write(f'"{file}_flip.png"')
            for seg in range(7):
                if mask % 3 == 1:
                    l.write(",0.5")
                elif mask % 3 == 2:
                    l.write(",1")
                else:
                    l.write(",0")
                mask //= 3
            l.write("\n")'''

'''for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/testing8/"):
    shutil.move(f"/home/wang-zhisheng/Downloads/centad/dataset/testing8/{file}",
                        f"/home/wang-zhisheng/Downloads/centad/dataset/training/{file}")'''

'''for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/training/"):
    if "new" not in file:
        shutil.move(f"/home/wang-zhisheng/Downloads/centad/dataset/training/{file}",
                    f"/home/wang-zhisheng/Downloads/centad/dataset/old/{file}")'''

'''with open("/home/wang-zhisheng/Downloads/centad/dataset/training_new.csv", "r") as f,\
        open("/home/wang-zhisheng/Downloads/centad/dataset/training.csv", "a") as f2:
    for line in f:
        filename = line.split(",")[0][1:-1]
        if "new" in filename:
            f2.write(line)'''

'''for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/testing10/"):
    mask = int(file.split("_")[-3])
    label = []
    for seg in range(7):
        if mask % 3 == 1:
            label.append(0.5)
        elif mask % 3 == 2:
            label.append(1)
        else:
            label.append(0)
        mask //= 3
    with open("/home/wang-zhisheng/Downloads/centad/dataset/testing10.csv", "a") as f:
        f.write(f'"{file}"')
        for x in label:
            f.write("," + str(x))
        f.write("\n")'''

'''for file in os.listdir("/home/wang-zhisheng/Downloads/centad/dataset/testing10/"):
    shutil.move(f"/home/wang-zhisheng/Downloads/centad/dataset/testing10/{file}",
                f"/home/wang-zhisheng/Downloads/centad/dataset/training/{file}")'''