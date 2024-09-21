import torch
import cv2
import os
from pathlib import Path

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/installed/extraction.pt/model')

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Directory containing test images
test_dir = 'directory/to/test/images'
# Directory to save extracted objects as images
output_dir = 'directory/to/save/annotated/pages'
# Directory to save extracted 
objects_dir = 'directory/to/save/extracted/images/as/separate/files'

# Create output directories if they don't already exist 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(objects_dir, exist_ok=True)

# Loop through all images in the test directory
for img_name in os.listdir(test_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_dir, img_name)
        
        # Read and preprocess the image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = model(img_rgb)
        
        # Process results
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        
        # Draw bounding boxes and extract each detected object
        for i, (box, score, cat) in enumerate(zip(boxes, scores, categories)):
            box = [int(x) for x in box]
            label = f"{model.names[int(cat)]}"
            # Green color for the box 
            color = (0, 255, 0)
            
            #Draw bounding box on the image
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            label_with_score = f"{label} {score:.2f}"
            cv2.putText(img, label_with_score, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            #Exrtact and save the object
            object_img = img_rgb[box[1]:box[3], box[0]:box[2]]
            object_filename = f"{Path(img_name).stem}_{label}_{i}.jpg"
            object_path = os.path.join(objects_dir, object_filename)
            cv2.imwrite(object_path, cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR))
            
        
        # Save the marked image
        output_path = os.path.join(output_dir, f"marked_{img_name}")
        cv2.imwrite(output_path, img)
        
        print(f"Processed {img_name}")

print("Processing completed.")
print(f"Marked images saved in '{output_dir}' directory.")
print(f"Extracted objects saved in '{objects_dir}' directory.")