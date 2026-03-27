from PIL import Image, ImageDraw
import pytesseract
import os

# Find tesseract
possible_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]

for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Using Tesseract at: {path}\n")
        break

# Load image
image = Image.open("lankadeepa_img.jpg")

# Get detailed OCR data with bounding boxes
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# Create visualization
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Process and visualize
print("Detected Text Blocks:\n")
block_count = 0

for i in range(len(data['text'])):
    confidence = int(data['conf'][i])
    text = data['text'][i].strip()
    
    # Only process confident, non-empty text
    if confidence > 30 and text:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Draw box
        draw.rectangle([x, y, x + w, y + h], outline='red', width=3)
        
        # Print info
        block_count += 1
        print(f"Block {block_count}:")
        print(f"  Text: {text}")
        print(f"  Position: ({x}, {y}), Size: {w}x{h}")
        print(f"  Confidence: {confidence}%\n")

# Save
image_with_boxes.save("detected_layout.jpg")
print(f"\nTotal blocks found: {block_count}")
print("Layout saved to 'detected_layout.jpg'")