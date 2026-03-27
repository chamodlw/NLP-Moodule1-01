from PIL import Image
import cv2
import numpy as np
import os

def separate_news_articles(image_path, output_folder="separated_articles", max_article_percent=50):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Calculate maximum article dimensions (50% of page)
    img_height, img_width = image.shape[:2]
    max_width = int(img_width * (max_article_percent / 100))
    max_height = int(img_height * (max_article_percent / 100))
    max_area = max_width * max_height
    
    print(f"Image size: {img_width}x{img_height}")
    print(f"Maximum article size: {max_width}x{max_height}")
    print(f"Maximum article area: {max_area} pixels\n")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image (white borders become white, content becomes black)
    # Adjust threshold value if needed (230-250 works well for white borders)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    
    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours (boundaries of each article)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (remove very small ones and very large ones)
    min_area = 1000  # Adjust based on your image size
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if contour meets size criteria
        if area > min_area and w <= max_width and h <= max_height and area <= max_area:
            valid_contours.append(cnt)
        elif area > min_area:
            print(f"Skipped oversized region: {w}x{h} (area: {area})")
    
    # Sort by position (top to bottom, left to right)
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    
    print(f"\nFound {len(valid_contours)} valid article sections\n")
    
    # Extract each article
    for idx, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add small margin (adjust as needed)
        margin = 10
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
        print(f"  Area: {w*h} pixels")
        print(f"  Saved to: {output_path}\n")
        
        # Draw rectangle on original for visualization
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # Add article number label
        cv2.putText(image, f"#{idx+1}", (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite(os.path.join(output_folder, "detected_layout2.jpg"), image)
    print(f"Layout visualization saved to {output_folder}/detected_layout2.jpg")

# Usage
separate_news_articles("lankadeepa_img2.jpg", max_article_percent=50)