#!/usr/bin/env python3
"""
Newspaper Article Separator v6 - Final Optimized Version
Specifically tuned for Sinhala newspaper layouts
"""

import cv2
import numpy as np
import os

def separate_news_articles(image_path, output_folder="separated_articles"):
    """
    Detect and separate news articles from newspaper image
    Optimized for Sinhala newspapers with complex layouts
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    original = image.copy()
    img_height, img_width = image.shape[:2]
    
    print(f"Image dimensions: {img_width} x {img_height} pixels")
    print(f"Total area: {img_width * img_height:,} pixels\n")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Applying detection algorithms...")
    
    # Simple thresholding to detect white borders (235+ is typically white border)
    _, white_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # Invert to get content areas
    content_mask = cv2.bitwise_not(white_mask)
    
    # Apply minimal morphological operations
    kernel = np.ones((2, 2), np.uint8)
    
    # Very light dilation to connect nearby text
    content_mask = cv2.dilate(content_mask, kernel, iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} initial contours\n")
    
    # Define filtering criteria
    min_area = 2500  # Minimum 50x50 pixels
    max_area = img_width * img_height * 0.45  # Maximum 45% of total page
    min_width = 50
    min_height = 50
    max_width = img_width * 0.95
    max_height = img_height * 0.95
    
    # Filter contours
    valid_boxes = []
    
    for idx, cnt in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Skip if too small
        if area < min_area:
            continue
        
        # Skip if too large
        if area > max_area:
            continue
        
        # Skip if dimensions are invalid
        if w < min_width or h < min_height:
            continue
        
        if w > max_width or h > max_height:
            continue
        
        # Check aspect ratio (width/height)
        aspect_ratio = w / h if h > 0 else 0
        
        # Skip extremely thin boxes
        if aspect_ratio < 0.05 or aspect_ratio > 20:
            continue
        
        # Calculate density of the region
        roi = content_mask[y:y+h, x:x+w]
        white_pixels = np.sum(roi == 255)
        density = white_pixels / area if area > 0 else 0
        
        # Skip regions with very low density (likely empty)
        if density < 0.02:
            continue
        
        valid_boxes.append({
            'box': [x, y, w, h],
            'area': area,
            'density': density
        })
    
    print(f"After size filtering: {len(valid_boxes)} candidates\n")
    
    # Remove overlapping boxes (keep the one with higher density)
    def boxes_overlap(box1, box2, threshold=0.5):
        """Check if two boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / min(box1_area, box2_area)
        
        return overlap_ratio > threshold
    
    # Sort by area (largest first)
    valid_boxes.sort(key=lambda x: x['area'], reverse=True)
    
    final_boxes = []
    used = [False] * len(valid_boxes)
    
    for i, box_info in enumerate(valid_boxes):
        if used[i]:
            continue
        
        # Check against already selected boxes
        overlaps = False
        for other_box in final_boxes:
            if boxes_overlap(box_info['box'], other_box, threshold=0.4):
                overlaps = True
                break
        
        if not overlaps:
            final_boxes.append(box_info['box'])
            used[i] = True
    
    print(f"After removing overlaps: {len(final_boxes)} articles\n")
    
    # Sort boxes by position (top to bottom, left to right)
    final_boxes.sort(key=lambda box: (box[1], box[0]))
    
    print("="*70)
    print(f"DETECTED {len(final_boxes)} NEWS ARTICLES")
    print("="*70 + "\n")
    
    # Save each article
    for idx, box in enumerate(final_boxes):
        x, y, w, h = box
        
        # Add small margin
        margin = 5
        x_crop = max(0, x - margin)
        y_crop = max(0, y - margin)
        w_crop = min(img_width - x_crop, w + 2 * margin)
        h_crop = min(img_height - y_crop, h + 2 * margin)
        
        # Extract article
        article = original[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        
        # Save to file
        article_filename = f"article_{idx+1:02d}.jpg"
        article_path = os.path.join(output_folder, article_filename)
        cv2.imwrite(article_path, article)
        
        print(f"Article #{idx+1:02d}")
        print(f"  Location: ({x}, {y})")
        print(f"  Size: {w} x {h} pixels")
        print(f"  Area: {w*h:,} pixels")
        print(f"  Saved: {article_filename}\n")
        
        # Draw on visualization image
        # Green box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Label background
        label_h = 22
        cv2.rectangle(image, (x, y - label_h), (x + 55, y), (0, 200, 0), -1)
        
        # Label text
        label_text = f"#{idx+1}"
        cv2.putText(image, label_text, (x + 3, y - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save visualization
    viz_path = os.path.join(output_folder, "detected_layout6.jpg")
    cv2.imwrite(viz_path, image)
    
    print("="*70)
    print(f"✓ Visualization saved: {viz_path}")
    print(f"✓ All articles saved to: {output_folder}/")
    print(f"✓ Total articles: {len(final_boxes)}")
    print("="*70)
    
    return len(final_boxes), final_boxes

# Main execution
if __name__ == "__main__":
    # Configuration
    input_image = "lankadeepa_img2.jpg"
    output_dir = "separated_articles"
    
    # Check if input exists
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image '{input_image}' not found!")
        print("Please make sure the image file is in the same directory.")
        exit(1)
    
    try:
        # Run detection
        num_articles, boxes = separate_news_articles(input_image, output_dir)
        
        print(f"\n✅ SUCCESS! Detected {num_articles} articles from the newspaper.")
        print(f"📁 Check the '{output_dir}' folder for individual articles.")
        print(f"🖼️  View 'detected_layout6.jpg' to see the detection results.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()