import torch
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Define augmentations using Albumentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.Rotate(limit=10),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0)),
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    train_images, val_images = train_test_split(image_files, train_size=train_ratio, random_state=42)

    for split, image_set in [('train', train_images), ('val', val_images)]:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
        for image_file in image_set:
            # Copy image file
            src_image = os.path.join(image_dir, image_file)
            dst_image = os.path.join(output_dir, 'images', split, image_file)
            shutil.copy(src_image, dst_image)
            
            # Try to copy label file
            label_file = image_file.replace('.jpg', '.txt')
            src_label = os.path.join(label_dir, label_file)
            dst_label = os.path.join(output_dir, 'labels', split, label_file)
            
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
            else:
                print(f"Warning: Label file not found for {image_file}. Skipping label.")

    print(f"Dataset split completed. Train set: {len(train_images)}, Validation set: {len(val_images)}")

def validate_and_normalize_yolo_annotation(annotation, image_width, image_height):
    class_id, x_center, y_center, width, height = annotation
    
    # Ensure all values are float
    try:
        class_id = int(class_id)
        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
    except ValueError:
        print(f"Warning: Non-numeric values in annotation: {annotation}")
        return None

    # Normalize values if they're not in range (0, 1]
    if x_center > 1 or y_center > 1 or width > 1 or height > 1:
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

    # Check if values are in the correct range
    if not all(0 < val <= 1 for val in [x_center, y_center, width, height]):
        print(f"Warning: Invalid YOLO format values: {annotation}")
        return None
    
    return [class_id, x_center, y_center, width, height]

def load_and_validate_annotations(label_path, image_width, image_height):
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return []

    with open(label_path, 'r') as file:
        annotations = file.readlines()
    
    valid_annotations = []
    for ann in annotations:
        parts = ann.strip().split()
        if len(parts) != 5:
            print(f"Warning: Invalid annotation format in {label_path}")
            continue
        
        validated_ann = validate_and_normalize_yolo_annotation(parts, image_width, image_height)
        if validated_ann:
            valid_annotations.append(validated_ann)
    
    return valid_annotations

def yolo_to_albumentations_bbox(bbox, image_width, image_height):
    # Convert from YOLO (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
    x_center, y_center, width, height = map(float, bbox[1:])
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    return [x_min, y_min, x_max, y_max]

def apply_augmentation(image, annotations, image_width, image_height):
    image_np = np.array(image)

    # Convert YOLO annotations to Albumentations format with labels (class_id as labels)
    albumentations_bboxes = [
        [x_center, y_center, width, height, class_id]  # Pass class_id as the label
        for class_id, x_center, y_center, width, height in annotations
    ]

    try:
        augmented = augmentations(image=image_np, bboxes=albumentations_bboxes, labels=[ann[0] for ann in annotations])
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']

        # Convert back to YOLO format, stripping out the labels
        yolo_bboxes = [
            [class_id, x_center, y_center, width, height]  # Restore class_id as first element
            for (x_center, y_center, width, height, class_id) in augmented_bboxes
        ]

        augmented_image_pil = Image.fromarray(augmented_image)
        
        return augmented_image_pil, yolo_bboxes
    except ValueError as e:
        print(f"Augmentation error: {str(e)}")
        return image, annotations

def save_annotations(label_path, annotations):
    with open(label_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')


def process_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=5):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get all image files from the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    for image_file in image_files:
        
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        
        # Load the image
        image = Image.open(image_path)
        image_width, image_height = image.size
        
        # Load and validate the annotations
        annotations = load_and_validate_annotations(label_path, image_width, image_height)
        
        if not annotations:
            print(f"Warning: No valid annotations for {image_file}. Skipping this image.")
            continue
        
        # Apply augmentations
        for i in range(num_augmentations):
            try:
                # Reload the original image for each augmentation
                image = Image.open(image_path)

                # Apply augmentations
                augmented_image, augmented_annotations = apply_augmentation(image, annotations, image_width, image_height)
                
                # Save the augmented image and annotations with unique names
                # Save with a unique name by combining the original image filename and augmentation number
                output_image_path = os.path.join(output_image_dir, f"{image_file.rsplit('.', 1)[0]}_aug{i}.jpg")
                augmented_image.save(output_image_path)
                
                # Save with a unique name by combining the original image filename and augmentation number
                output_label_path = os.path.join(output_label_dir, f"{image_file.rsplit('.', 1)[0]}_aug{i}.txt")
                save_annotations(output_label_path, augmented_annotations)
                
                print(f"Processed and saved {output_image_path} and its annotations")
            except Exception as e:
                print(f"Error processing {image_file} (augmentation {i}): {str(e)}")

    print(f"Processed {len(image_files)} images with {num_augmentations} augmentations each.")

def main():
    # Define paths
    image_dir = './dataset3/images/'
    label_dir = './dataset3/labels/'
    output_dir = './processed_dataset/'
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split dataset into train and val (80% train, 20% val)
    split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8)

    # Process and augment all the images in the training dataset
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    output_images_dir = os.path.join(output_dir, 'augmented_images_train')
    output_labels_dir = os.path.join(output_dir, 'augmented_labels_train')
    
    process_dataset(train_images_dir, train_labels_dir, output_images_dir, output_labels_dir, num_augmentations=5)
    
    print("Training data augmentation completed.")
    print("Validation set is not augmented to maintain data integrity.")

if __name__ == "__main__":
    main()