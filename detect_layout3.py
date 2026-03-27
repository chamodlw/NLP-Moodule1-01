from PIL import Image
import cv2
import numpy as np
import os

def separate_news_articles_improved(image_path, output_folder="separated_articles"):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    original = image.copy()
    height, width = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better border detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Alternative: Try simple threshold with lower value
    # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to connect components
    # Horizontal kernel to connect text in same article
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    
    # Vertical kernel to connect vertically
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    
    # Combine horizontal and vertical
    combined = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
    
    # Dilate to merge nearby regions
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    min_area = width * height * 0.01  # At least 1% of image area
    max_area = width * height * 0.8   # At most 80% of image area
    min_width = width * 0.1           # At least 10% of image width
    min_height = height * 0.05        # At least 5% of image height
    
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter based on size criteria
        if (min_area < area < max_area and 
            w > min_width and 
            h > min_height):
            valid_contours.append(cnt)
    
    # Sort by position (top to bottom, left to right)
    valid_contours = sorted(valid_contours, 
                           key=lambda c: (cv2.boundingRect(c)[1] // 100, 
                                         cv2.boundingRect(c)[0]))
    
    print(f"Found {len(valid_contours)} article sections\n")
    
    # Save intermediate images for debugging
    cv2.imwrite(os.path.join(output_folder, "1_binary.jpg"), binary)
    cv2.imwrite(os.path.join(output_folder, "2_horizontal.jpg"), horizontal)
    cv2.imwrite(os.path.join(output_folder, "3_vertical.jpg"), vertical)
    cv2.imwrite(os.path.join(output_folder, "4_dilated.jpg"), dilated)
    
    # Extract each article
    for idx, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add margin
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(original.shape[1] - x, w + 2 * margin)
        h = min(original.shape[0] - y, h + 2 * margin)
        
        # Crop article
        article = original[y:y+h, x:x+w]
        
        # Save
        output_path = os.path.join(output_folder, f"article_{idx+1}.jpg")
        cv2.imwrite(output_path, article)
        
        print(f"Article {idx+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Saved to: {output_path}\n")
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(image, str(idx+1), (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(os.path.join(output_folder, "detected_layout3.jpg"), image)
    print(f"\nLayout visualization saved to {output_folder}/detected_layout3.jpg")
    print(f"Check intermediate images (1_binary.jpg through 4_dilated.jpg) to debug")

# Usage
separate_news_articles_improved("lankadeepa_img.jpg")