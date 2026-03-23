import cv2
from pathlib import Path
import numpy as np
import os

INPUT_DIR = "CHANGE_INPUT_DIR"
OUTPUT_DIR = "CHANGE_OUTPUT_DIR"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def despeckle_bw(image_path, output_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return
    
    denoised = cv2.fastNlMeansDenoising(img, None, h=12, templateWindowSize=7, searchWindowSize=21)

    adaptive_mask = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        65, 5
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive_mask, 8)
    filtered_mask = np.zeros_like(adaptive_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        corners = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(corners)
        length    = max(rw, rh)
        thickness = min(rw, rh)

        is_dot = (thickness < 20) and (length < 12)
        is_thin_line = (thickness < 2) and (length > 12)

        if is_dot or is_thin_line:
            filtered_mask[labels == i] = 255

    filtered_mask = cv2.dilate(filtered_mask, np.ones((4, 4), np.uint8), iterations=1)

    _, paper_limit = cv2.threshold(denoised, 240, 255, cv2.THRESH_BINARY_INV)
    final_mask = cv2.bitwise_and(filtered_mask, paper_limit)

    inpaint_mask = (final_mask == 255).astype(np.uint8) * 255
    result = cv2.inpaint(
        denoised,
        inpaint_mask,
        3,
        cv2.INPAINT_TELEA
    )

    denoised2 = cv2.fastNlMeansDenoising(result, None, h=12, templateWindowSize=7, searchWindowSize=21)

    cv2.imwrite(str(output_path), denoised2)


for file in Path(INPUT_DIR).glob("*.png"):
    despeckle_bw(file, Path(OUTPUT_DIR) / file.name)
    print(f"Done: {file.name}")
