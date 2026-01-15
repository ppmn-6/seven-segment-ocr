import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

dir = "/home/wang-zhisheng/Downloads/centad/dataset/testing5/"
files = [f for f in os.listdir(dir) if os.path.isfile(dir + f)]

# Scale patch size, kernel size, blur radius with image dimensions accordingly

def off(i):
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = np.array(points[i], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    bg[mask == 255] = 255

def dim(i):
    mask = np.zeros(img.shape, dtype=np.uint8)
    polygon = np.array(points[i], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel)
    smooth = cv2.GaussianBlur(mask.astype(np.float32)/255, (21, 21), 0)
    out = result.astype(np.float32)
    bright = (img.astype(np.float32) * random.uniform(0.6, 0.75))

    output = out * (1 - smooth) + bright * smooth
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

for file in files:
    file = "Screenshot From 2026-01-05 04-26-11.png"
    metadata = dir + "metadata/" + file[:-3] + "txt"
    points = []
    with open(metadata, "r") as f:
        lines = f.readlines()
        pos = 0
        for seg in range(7):
            cnt = int(lines[pos])
            points.append([])
            for i in range(cnt):
                pos += 1
                x = int(lines[pos].split()[0])
                y = int(lines[pos].split()[1])
                points[-1].append((x, y))
            pos += 1
    

    img = cv2.imread(dir + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    bg = np.full(img.shape, 255, dtype=np.uint8)
    for i in range(7):
        polygon = np.array(points[i], dtype=np.int32)
        cv2.fillPoly(bg, [polygon], 0)
    ori_bg = bg.copy()

    patches = []
    h, w = img.shape
    patch_size = 3
    for y in range(0, h - patch_size + 1, patch_size // 2):
        for x in range(0, w - patch_size + 1, patch_size // 2):
            patch_mask = bg[y:y+patch_size, x:x+patch_size]
            if np.all(patch_mask == 255):  # only full-source patches
                patches.append(img[y:y+patch_size, x:x+patch_size])
    
    # Exclude:
    # digits with <=2 existent segments (except the digit 1)
    # nonsensical binary digits
    # digits with no completely lit segment
    for mask in range(3 ** 7):
        mask2 = mask
        isBinary = True
        isLit = False
        bin = 0
        count = 0
        label = []
        for seg in range(7):
            if mask2 % 3 == 1:
                isBinary = False
                count += 1
                label.append(0.5)
            elif mask2 % 3 == 2:
                isLit = True
                bin |= 1 << seg
                count += 1
                label.append(1)
            else:
                label.append(0)
            mask2 //= 3
        flag = True
        if count <= 2 or not isLit:
            flag = False
        digits = (0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101,
                        0b1111101, 0b0000111, 0b1111111, 0b1101111)
        if isBinary and bin not in digits:
            flag = False
        if mask == 24:
            flag = True
        if not flag:
            continue
        
        for i in range(3):
            bg = ori_bg.copy()
            mask2 = mask
            result = img.copy()
            for seg in range(7):
                if mask2 % 3 == 0:
                    off(seg)
                elif mask2 % 3 == 1:
                    result = dim(seg)
                    isBinary = False
                else:
                    bin |= 1 << seg
                mask2 //= 3
            
            h, w = img.shape
            for y in range(0, h, patch_size // 2):
                for x in range(0, w, patch_size // 2):
                    if bg[y, x] == 255:  # pixel needs filling
                        patch = patches[np.random.randint(len(patches))]
                        y1, y2 = y, min(y + patch_size, h)
                        x1, x2 = x, min(x + patch_size, w)
                        
                        # Existing content in target
                        existing = result[y1:y2, x1:x2]
                        patch = patch[:y2-y1, :x2-x1]

                        # Blend if overlap exists
                        alpha = 1.0
                        patch_mask = bg[y:y+patch_size, x:x+patch_size]
                        if np.all(patch_mask == 255):
                            result[y1:y2, x1:x2] = cv2.addWeighted(existing, 1-alpha, patch, alpha, 0)
            
            blur = cv2.GaussianBlur(result, (7, 7), 0)
            result[bg == 255] = blur[bg == 255]
            cv2.imwrite(f"/home/wang-zhisheng/Downloads/centad/dataset/testing11/{file}_{mask}_{i}.png", result)
    break

'''            filename = f"{file}_{mask}_{i}.png"
            with open("/home/wang-zhisheng/Downloads/centad/dataset/testing4.csv", "a") as f:
                f.write(f'"{filename}"')
                for l in label:
                    f.write("," + str(l))
                f.write("\n")
            cv2.imwrite(f"/home/wang-zhisheng/Downloads/centad/dataset/testing4/{filename}", result)'''
