import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the newspaper image
image = cv2.imread('lankadeepa_img.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load a pre-trained layout detection model
# This model is trained on newspaper/magazine layouts
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Detect layout elements
layout = model.detect(image_rgb)

# Filter and process detected blocks
blocks = []
for idx, block in enumerate(layout):
    # Get coordinates
    x1, y1, x2, y2 = int(block.x_1), int(block.y_1), int(block.x_2), int(block.y_2)
    
    # Add margin (10-15px as you mentioned)
    margin = 12
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)
    
    blocks.append({
        'bbox': (x1, y1, x2, y2),
        'type': block.type,
        'score': block.score
    })

# Save individual articles
for idx, block in enumerate(blocks):
    x1, y1, x2, y2 = block['bbox']
    cropped = image_rgb[y1:y2, x1:x2]
    
    # Save the cropped article
    Image.fromarray(cropped).save(f'article_{idx}_{block["type"]}.jpg')
    print(f"Saved article_{idx}_{block['type']}.jpg - Size: {x2-x1}x{y2-y1}")

# Visualize the detected layout
lp.draw_box(image_rgb, layout, box_width=3)
plt.figure(figsize=(15, 20))
plt.imshow(image_rgb)
plt.axis('off')
plt.savefig('detected_layout4.jpg', bbox_inches='tight', dpi=150)
plt.show()

print(f"\nTotal articles detected: {len(blocks)}")