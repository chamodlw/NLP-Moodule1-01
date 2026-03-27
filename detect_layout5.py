import cv2
import numpy as np
import os
import fitz  # PyMuPDF
from PIL import Image

def separate_news_articles_from_pdf(pdf_path, output_folder="separated_articles", dpi=300, debug=False):
    """
    Extract individual news articles from a newspaper PDF.
    Uses multiple detection strategies for better article separation.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Output directory for extracted articles
        dpi: Resolution for PDF to image conversion (higher = better quality)
        debug: If True, saves intermediate processing images for debugging
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    if debug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)
    
    print(f"Opening PDF: {pdf_path}")
    print(f"Converting PDF to images (DPI: {dpi})...")
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        print(f"Found {num_pages} page(s) in PDF\n")
        
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return
    
    article_counter = 1
    
    # Process each page
    for page_num in range(num_pages):
        print(f"=" * 60)
        print(f"Processing Page {page_num + 1}/{num_pages}")
        print(f"=" * 60)
        
        # Get page
        page = pdf_document[page_num]
        
        # Calculate zoom factor for desired DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to image
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGB to BGR for OpenCV
        if pix.n == 4:  # RGBA
            image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        original = image.copy()
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # METHOD 1: Detect white space separators (common in newspapers)
        # Invert: white becomes black, content becomes white
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        if debug:
            cv2.imwrite(os.path.join(debug_folder, f"page{page_num+1}_01_binary.jpg"), binary)
        
        # METHOD 2: Detect lines and borders
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        detect_horizontal = cv2.morphologyEx(~gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        detect_vertical = cv2.morphologyEx(~gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        lines = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)
        
        if debug:
            cv2.imwrite(os.path.join(debug_folder, f"page{page_num+1}_02_lines.jpg"), lines)
        
        # METHOD 3: Text block detection
        # Adaptive threshold to detect text
        text_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Connect nearby characters into words
        kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        text_blocks = cv2.morphologyEx(text_binary, cv2.MORPH_CLOSE, kernel_word)
        
        # Connect words into lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        text_blocks = cv2.morphologyEx(text_blocks, cv2.MORPH_CLOSE, kernel_line)
        
        # Connect lines into paragraphs
        kernel_para = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
        text_blocks = cv2.morphologyEx(text_blocks, cv2.MORPH_CLOSE, kernel_para, iterations=2)
        
        if debug:
            cv2.imwrite(os.path.join(debug_folder, f"page{page_num+1}_03_text_blocks.jpg"), text_blocks)
        
        # Find contours from text blocks
        contours, hierarchy = cv2.findContours(text_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate minimum area (smaller threshold for better detection)
        page_area = height * width
        min_area = page_area * 0.0001  # 0.1% of page
        max_area = page_area * 0.4    # 40% of page (avoid detecting entire page)
        
        # Filter and classify contours
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                
                # Filter by aspect ratio and minimum size
                if 0.15 < aspect_ratio < 8 and w > 150 and h > 150:
                    # Check if not mostly white space
                    roi = text_binary[y:y+h, x:x+w]
                    white_pixels = np.sum(roi == 255)
                    total_pixels = w * h
                    text_density = white_pixels / total_pixels
                    
                    # Only keep if has sufficient text content
                    if text_density > 0.05:  # At least 5% text
                        valid_contours.append(cnt)
        
        print(f"Initial detection: {len(valid_contours)} potential articles")
        
        # Remove overlapping contours (keep larger ones)
        valid_contours = remove_overlapping_contours(valid_contours, overlap_threshold=0.5)
        
        print(f"After removing overlaps: {len(valid_contours)} articles")
        
        # Sort contours by position (top to bottom, left to right)
        # Group by rows first, then sort within rows
        valid_contours = sort_contours_newspaper_layout(valid_contours)
        
        print(f"\nFound {len(valid_contours)} article section(s) on page {page_num + 1}\n")
        
        # Save page layout visualization
        page_viz = original.copy()
        
        # Extract each article
        for idx, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add margin
            margin = 15
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(width - x, w + 2 * margin)
            h = min(height - y, h + 2 * margin)
            
            # Crop article
            article = original[y:y+h, x:x+w]
            
            # Save with global counter
            output_path = os.path.join(output_folder, f"article_{article_counter:03d}_page{page_num+1}.jpg")
            cv2.imwrite(output_path, article, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"Article {article_counter} (Page {page_num+1}, Section {idx+1}):")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {w}x{h} pixels")
            print(f"  Area: {w*h} pixels²")
            print(f"  Saved to: {output_path}\n")
            
            # Draw rectangle on visualization
            color = (0, 255, 0)  # Green
            cv2.rectangle(page_viz, (x, y), (x+w, y+h), color, 4)
            
            # Add article number
            label = str(article_counter)
            font_scale = 1.2
            thickness = 3
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background for text
            cv2.rectangle(page_viz, (x+5, y+5), (x+15+label_w, y+15+label_h), (0, 0, 255), -1)
            cv2.putText(page_viz, label, (x+10, y+10+label_h), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            article_counter += 1
        
        # Save page visualization
        viz_path = os.path.join(output_folder, f"page_{page_num+1}_layout.jpg")
        cv2.imwrite(viz_path, page_viz, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Page {page_num+1} layout visualization saved to: {viz_path}\n")
    
    # Close PDF
    pdf_document.close()
    
    print("=" * 60)
    print(f"Extraction complete! Total articles extracted: {article_counter - 1}")
    print(f"All files saved to: {output_folder}")
    print("=" * 60)


def remove_overlapping_contours(contours, overlap_threshold=0.5):
    """Remove overlapping contours, keeping the larger ones."""
    if len(contours) == 0:
        return []
    
    # Get bounding boxes
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    areas = [w * h for x, y, w, h in boxes]
    
    # Sort by area (descending)
    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    for i in sorted_indices:
        x1, y1, w1, h1 = boxes[i]
        
        # Check overlap with kept contours
        overlap = False
        for j in keep:
            x2, y2, w2, h2 = boxes[j]
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                smaller_area = min(w1 * h1, w2 * h2)
                
                if intersection / smaller_area > overlap_threshold:
                    overlap = True
                    break
        
        if not overlap:
            keep.append(i)
    
    return [contours[i] for i in sorted(keep)]


def sort_contours_newspaper_layout(contours, tolerance=50):
    """
    Sort contours in newspaper reading order (top to bottom, left to right within rows).
    tolerance: vertical tolerance for grouping contours into rows
    """
    if len(contours) == 0:
        return []
    
    # Get bounding boxes
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    # Group into rows based on y-coordinate
    rows = []
    for i, (x, y, w, h) in enumerate(boxes):
        center_y = y + h // 2
        
        # Find matching row
        placed = False
        for row in rows:
            row_y = row[0][1]  # y-coordinate of first box in row
            if abs(center_y - row_y) < tolerance:
                row.append((i, center_y, x))
                placed = True
                break
        
        if not placed:
            rows.append([(i, center_y, x)])
    
    # Sort rows by y-coordinate
    rows.sort(key=lambda row: row[0][1])
    
    # Sort within each row by x-coordinate
    sorted_indices = []
    for row in rows:
        row.sort(key=lambda item: item[2])  # Sort by x
        sorted_indices.extend([item[0] for item in row])
    
    return [contours[i] for i in sorted_indices]


def separate_news_articles_from_image(image_path, output_folder="separated_articles", debug=False):
    """
    Extract individual news articles from a newspaper image.
    """
    os.makedirs(output_folder, exist_ok=True)
    if debug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    original = image.copy()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Text block detection
    text_binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    text_blocks = cv2.morphologyEx(text_binary, cv2.MORPH_CLOSE, kernel_word)
    
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    text_blocks = cv2.morphologyEx(text_blocks, cv2.MORPH_CLOSE, kernel_line)
    
    kernel_para = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
    text_blocks = cv2.morphologyEx(text_blocks, cv2.MORPH_CLOSE, kernel_para, iterations=2)
    
    contours, _ = cv2.findContours(text_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    page_area = height * width
    min_area = page_area * 0.001
    max_area = page_area * 0.4
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.15 < aspect_ratio < 8 and w > 150 and h > 150:
                roi = text_binary[y:y+h, x:x+w]
                white_pixels = np.sum(roi == 255)
                text_density = white_pixels / (w * h)
                if text_density > 0.05:
                    valid_contours.append(cnt)
    
    valid_contours = remove_overlapping_contours(valid_contours, overlap_threshold=0.5)
    valid_contours = sort_contours_newspaper_layout(valid_contours)
    
    print(f"Found {len(valid_contours)} article section(s)\n")
    
    for idx, contour in enumerate(valid_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        
        article = original[y:y+h, x:x+w]
        
        output_path = os.path.join(output_folder, f"article_{idx:03d}.jpg")
        cv2.imwrite(output_path, article, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"Article {idx}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Saved to: {output_path}\n")
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(image, str(idx), (x+10, y+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    viz_path = os.path.join(output_folder, "detected_layout5.jpg")
    cv2.imwrite(viz_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Layout visualization saved to: {viz_path}")


# Usage
if __name__ == "__main__":
    # For PDF files - with debug mode to see intermediate processing
    separate_news_articles_from_pdf(
        "newspaper.pdf", 
        output_folder="extracted_articles", 
        dpi=300,
        debug=True  # Set to False to disable debug images
    )
    
    # For image files
    # separate_news_articles_from_image("lankadeepa_img.jpg", output_folder="extracted_articles", debug=True)