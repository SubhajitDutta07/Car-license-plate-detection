import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from src.ocr_processor import OCRProcessor

class LicensePlateDetector:
    def __init__(self, model_path='models/best_license_plate_model.pt'):
        """
        Initialize license plate detector with OCR
        """
        try:
            self.model = YOLO(model_path)
            self.ocr_processor = OCRProcessor(ocr_method='easyocr')
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_with_ocr(self, img_path, display=True, save_path=None):
        """
        Predict license plates and perform OCR
        """
        try:
            # Run inference
            results = self.model.predict(img_path, conf=0.25)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image from {img_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_display = img_rgb.copy()
            
            detected_plates = []
            
            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Perform OCR on license plate
                    plate_text = self.ocr_processor.process_license_plate(img_rgb, (x1, y1, x2, y2))
                    
                    # Store detection
                    detected_plates.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'plate_text': plate_text
                    })
                    
                    # Draw bounding box and text
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Create label with confidence and plate text
                    label = f"{confidence*100:.1f}%"
                    if plate_text and plate_text != "Invalid Format":
                        label = f"{plate_text} | {label}"
                    
                    # Calculate text position
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        img_display,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        img_display,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2
                    )
            
            # Save image if requested
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            
            # Display if requested
            if display:
                plt.figure(figsize=(12, 8))
                plt.imshow(img_display)
                plt.axis('off')
                plt.title(f"Detected License Plates: {len(detected_plates)}")
                plt.show()
            
            return detected_plates, img_display
            
        except Exception as e:
            print(f"Error in predict_with_ocr: {e}")
            raise
    
    def process_video(self, video_path, output_path=None, display=False):
        """
        Process video for license plate detection
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize video writer if output path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            all_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                results = self.model.predict(frame, conf=0.25)
                
                frame_detections = []
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Perform OCR
                        plate_text = self.ocr_processor.process_license_plate(
                            frame, (x1, y1, x2, y2)
                        )
                        
                        frame_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'plate_text': plate_text,
                            'frame': frame_count
                        })
                        
                        # Draw on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{confidence*100:.1f}%"
                        if plate_text and plate_text != "Invalid Format":
                            label = f"{plate_text} | {label}"
                        
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                
                all_detections.extend(frame_detections)
                
                # Write frame to output video
                if output_path:
                    out.write(frame)
                
                # Display frame if requested
                if display and frame_count % 30 == 0:  # Display every 30th frame
                    cv2.imshow('License Plate Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
            
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"Processed {frame_count} frames")
            return all_detections
            
        except Exception as e:
            print(f"Error in process_video: {e}")
            raise