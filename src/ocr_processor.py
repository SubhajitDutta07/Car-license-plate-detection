import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import re

class OCRProcessor:
    def __init__(self, ocr_method='easyocr', languages=['en']):
        """
        Initialize OCR processor with chosen method
        """
        self.ocr_method = ocr_method
        self.languages = languages
        
        if ocr_method == 'easyocr':
            try:
                self.reader = easyocr.Reader(languages)
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                print("Falling back to Tesseract")
                self.ocr_method = 'tesseract'
        
        elif ocr_method == 'tesseract':
            try:
                # Configure tesseract path if needed
                # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                pass
            except Exception as e:
                print(f"Error configuring Tesseract: {e}")
    
    def preprocess_plate_image(self, image):
        """
        Preprocess license plate image for better OCR results
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Apply denoising
            thresh = cv2.medianBlur(thresh, 3)
            
            return thresh
            
        except Exception as e:
            print(f"Error in preprocess_plate_image: {e}")
            return image
    
    def extract_text(self, image):
        """
        Extract text from license plate image
        """
        try:
            if self.ocr_method == 'easyocr':
                # Use EasyOCR
                result = self.reader.readtext(image, detail=0)
                text = ' '.join(result) if result else ""
                
            elif self.ocr_method == 'tesseract':
                # Use Tesseract
                pil_image = Image.fromarray(image)
                text = pytesseract.image_to_string(pil_image, config='--psm 8 --oem 3')
                text = text.strip()
            
            else:
                raise ValueError(f"Unsupported OCR method: {self.ocr_method}")
            
            # Clean and validate license plate text
            cleaned_text = self.clean_license_plate_text(text)
            return cleaned_text
            
        except Exception as e:
            print(f"Error in extract_text: {e}")
            return ""
    
    def clean_license_plate_text(self, text):
        """
        Clean and format license plate text
        """
        if not text:
            return ""
        
        # Remove special characters and extra spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s-]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        # Validate format (basic validation)
        if len(cleaned) < 2 or len(cleaned) > 12:
            cleaned = "Invalid Format"
        
        return cleaned
    
    def process_license_plate(self, image, bbox):
        """
        Process license plate from bounding box
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract plate region
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return ""
            
            # Preprocess for OCR
            processed_plate = self.preprocess_plate_image(plate_region)
            
            # Extract text
            plate_text = self.extract_text(processed_plate)
            
            return plate_text
            
        except Exception as e:
            print(f"Error processing license plate: {e}")
            return ""