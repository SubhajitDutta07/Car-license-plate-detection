import os
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
import re

def number_in_string(filename):
    """
    Extract numbers from a filename for sorting
    """
    try:
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(0))
        else:
            return -1
    except Exception as e:
        print(f"Error extracting number from filename {filename}: {e}")
        return -1

def load_and_prepare_data(dataset_path):
    """
    Load XML annotations and prepare data for YOLO format
    """
    try:
        label_dict = dict(
            img_path=[],
            xmin=[],
            xmax=[],
            ymin=[],
            ymax=[],
            img_w=[],
            img_h=[]
        )
        
        xml_files = glob(f'{dataset_path}/annotations/*.xml')
        
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {dataset_path}/annotations/")
        
        for filename in sorted(xml_files, key=number_in_string):
            try:
                info = xet.parse(filename)
                root = info.getroot()
                
                member_object = root.find('object')
                if member_object is None:
                    print(f"No object found in {filename}, skipping...")
                    continue
                    
                label_info = member_object.find('bndbox')
                
                xmin = int(label_info.find('xmin').text)
                xmax = int(label_info.find('xmax').text)
                ymin = int(label_info.find('ymin').text)
                ymax = int(label_info.find('ymax').text)
                
                img_name = root.find('filename').text
                img_path = os.path.join(dataset_path, 'images', img_name)
                
                # Check if image exists
                if not os.path.exists(img_path):
                    print(f"Image {img_path} not found, skipping...")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image {img_path}, skipping...")
                    continue
                    
                height, width, _ = img.shape
                
                label_dict['img_path'].append(img_path)
                label_dict['xmin'].append(xmin)
                label_dict['xmax'].append(xmax)
                label_dict['ymin'].append(ymin)
                label_dict['ymax'].append(ymax)
                label_dict['img_w'].append(width)
                label_dict['img_h'].append(height)
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
        
        if not label_dict['img_path']:
            raise ValueError("No valid data loaded from XML files")
            
        alldata = pd.DataFrame(label_dict)
        return alldata
        
    except Exception as e:
        print(f"Error in load_and_prepare_data: {e}")
        raise

def create_yolo_dataset(data, base_output_dir='datasets/cars_license_plate_new'):
    """
    Split data and create YOLO format dataset
    """
    try:
        # Create train/val/test splits
        train, test = train_test_split(data, test_size=0.1, random_state=42)
        train, val = train_test_split(train, train_size=8/9, random_state=42)
        
        print(f"Dataset sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Create directories and convert to YOLO format
        make_split_dir_in_yolo_format('train', train, base_output_dir)
        make_split_dir_in_yolo_format('val', val, base_output_dir)
        make_split_dir_in_yolo_format('test', test, base_output_dir)
        
        return train, val, test
        
    except Exception as e:
        print(f"Error in create_yolo_dataset: {e}")
        raise

def make_split_dir_in_yolo_format(split_name, split_df, base_dir='datasets/cars_license_plate_new'):
    """
    Create YOLO format directories and files
    """
    try:
        labels_path = os.path.join(base_dir, split_name, 'labels')
        images_path = os.path.join(base_dir, split_name, 'images')
        
        # Create directories
        os.makedirs(labels_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        
        for _, row in split_df.iterrows():
            try:
                img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))
                
                # Convert to YOLO format
                x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
                y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
                width = (row['xmax'] - row['xmin']) / row['img_w']
                height = (row['ymax'] - row['ymin']) / row['img_h']
                
                label_path = os.path.join(labels_path, f'{img_name}.txt')
                with open(label_path, 'w') as file:
                    file.write(f'0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n')
                
                # Copy image
                shutil.copy(row['img_path'], os.path.join(images_path, img_name + img_extension))
                
            except Exception as e:
                print(f"Error processing row {_}: {e}")
                continue
        
        print(f'Created {images_path} and {labels_path}')
        
    except Exception as e:
        print(f"Error in make_split_dir_in_yolo_format: {e}")
        raise