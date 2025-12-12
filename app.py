import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

class LicensePlateDetector:
    def __init__(self):
        """Initialize the detector with your model.pt"""
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model from models/model.pt"""
        try:
            model_path = "models/model.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                st.sidebar.success("✅ Model loaded successfully")
            else:
                st.sidebar.error(f"❌ Model not found at: {model_path}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
    
    def detect_in_image(self, image_path, output_path):
        """Detect license plates in image"""
        try:
            # Run detection
            results = self.model.predict(image_path, conf=0.25, imgsz=640)
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                st.error("Could not read image")
                return output_path, []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add confidence text
                    label = f"{conf*100:.1f}%"
                    cv2.putText(img_rgb, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save output to temp
            cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            return output_path, detections
            
        except Exception as e:
            st.error(f"Image detection error: {str(e)}")
            return None, []
    
    def detect_in_video(self, video_path, output_path):
        """Detect license plates in video"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Cannot open video file")
                return None, []
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            detections = []
            frame_count = 0
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress
                if frame_count % 30 == 0:
                    progress_bar.progress(min(frame_count / 900, 1.0))
                    status_text.text(f"Processing frame {frame_count}")
                
                # Process every 3rd frame for speed
                if frame_count % 3 == 0:
                    results = self.model.predict(frame, conf=0.25, imgsz=640)
                    
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'frame': frame_count
                            })
                            
                            # Draw on frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{conf*100:.1f}%", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write frame
                out.write(frame)
            
            # Cleanup
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
            
            return output_path, detections
            
        except Exception as e:
            st.error(f"Video detection error: {str(e)}")
            return None, []
    
    def process_file(self, input_path):
        """Process image or video file"""
        # Create temp output file
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video output
            temp_output = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.mp4',
                dir='temp'
            )
            output_path = temp_output.name
            temp_output.close()
            
            return self.detect_in_video(input_path, output_path)
        
        else:
            # Image output
            temp_output = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.jpg',
                dir='temp'
            )
            output_path = temp_output.name
            temp_output.close()
            
            return self.detect_in_image(input_path, output_path)

def main():
    """Main Streamlit application"""
    st.title("🚗 License Plate Detection")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    if not detector.model:
        st.error("Model not loaded. Please ensure 'model.pt' is in the 'models/' directory")
        return
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence threshold for license plate detection"
        )
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Upload image or video")
        st.markdown("2. Click 'Detect' button")
        st.markdown("3. View results and download")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'mkv'],
        help="Upload file containing vehicles with license plates"
    )
    
    if uploaded_file is not None:
        # Determine file type
        is_video = uploaded_file.type.startswith('video') or \
                  uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if is_video:
                st.subheader("Original Video")
                video_bytes = uploaded_file.read()
                st.video(video_bytes)
                uploaded_file.seek(0)  # Reset for processing
            else:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
        
        # Detect button
        if st.button("🔍 Detect License Plates", type="primary"):
            with st.spinner("Processing..."):
                # Save uploaded file to temp
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1],
                    dir='temp'
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    input_path = tmp_file.name
                
                try:
                    # Process the file
                    result_path, detections = detector.process_file(input_path)
                    
                    # Display results
                    with col2:
                        if result_path and os.path.exists(result_path):
                            if is_video:
                                st.subheader("Detected License Plates")
                                with open(result_path, 'rb') as f:
                                    st.video(f.read())
                            else:
                                st.subheader("Detected License Plates")
                                result_img = Image.open(result_path)
                                st.image(result_img, use_column_width=True)
                    
                    # Show detection summary
                    if detections:
                        pass
                        
                        if is_video:
                            frames = len(set(d.get('frame', 0) for d in detections))
                            st.info(f"Detections occurred in {frames} frames")
                        
                        # Show detection details
                        with st.expander("View detection details"):
                            for i, det in enumerate(detections[:10]):  # Show first 10
                                st.write(f"Detection {i+1}: Confidence: {det['confidence']*100:.1f}%")
                    else:
                        st.warning("No license plates detected")
                    
                    # Download button
                    if result_path and os.path.exists(result_path):
                        with open(result_path, 'rb') as f:
                            file_data = f.read()
                        
                        download_name = f"detected_{uploaded_file.name}"
                        st.download_button(
                            label="📥 Download Processed File",
                            data=file_data,
                            file_name=download_name,
                            mime="application/octet-stream"
                        )
                
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                
                finally:
                    # Cleanup temp files
                    try:
                        if os.path.exists(input_path):
                            os.unlink(input_path)
                    except:
                        pass

if __name__ == "__main__":
    main()