import base64
import io
import re
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional, Dict
from fastapi import FastAPI
from PIL import Image
import pytesseract
from ultralytics import YOLO
from dateutil.parser import parse

# --- Models & Configuration ---

app = FastAPI()

# Load the YOLO model once on startup
# This path is correct because the Dockerfile copies the 'models' folder to '/app/models'
MODEL_PATH = "/app/models/best(1).pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Regex to find dates (e.g., 12/25/2024, 25 DEC 2024)
DATE_REGEX = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} (JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC) \d{2,4})'

# Base shelf life in days (a simple estimation dictionary)
BASE_SHELF_LIFE_DAYS = {
    'milk': 7,
    'chicken': 2,
    'egg': 21,
    'lettuce': 10,
    'yogurt': 14,
    'default': 5  # Default for items we don't know
}

# --- Pydantic Models (for API data validation) ---

class SensorData(BaseModel):
    temperature_c: Optional[float] = Field(None, alias="temperature_c")
    humidity_percent: Optional[float] = Field(None, alias="humidity_percent")
    gas_level_percent: Optional[float] = Field(None, alias="gas_level_percent")

class ImageData(BaseModel):
    image_base64: str
    sensor_data: Optional[SensorData] = None

class DetectedItem(BaseModel):
    name: str
    expiration_date: Optional[str] = None
    confidence: float
    is_approximate: bool = False

class VisionData(BaseModel):
    detected_items: list[DetectedItem]
    raw_ocr_text: str

# --- Helper Functions ---

def find_expiration_date(text: str) -> Optional[str]:
    """Parses text to find the most likely expiration date."""
    try:
        matches = re.findall(DATE_REGEX, text, re.IGNORECASE)
        if not matches:
            return None
        
        # Parse the first found date
        # matches[0] might be a tuple, e.g., ('12/25/2024', '')
        date_str = matches[0][0] 
        
        # Use dateutil.parser to smartly parse the date
        parsed_date = parse(date_str)
        return parsed_date.isoformat()

    except Exception as e:
        print(f"⚠️ Error parsing date: {e}")
        return None

def approximate_expiration(label: str, sensors: Optional[SensorData]) -> str:
    """
    Approximates expiration date based on item label and sensor data.
    """
    now = datetime.now()
    
    # 1. Get base shelf life
    base_days = BASE_SHELF_LIFE_DAYS.get(label.lower(), BASE_SHELF_LIFE_DAYS['default'])
    
    # 2. Adjust shelf life based on sensors
    adjustment_factor = 1.0  # Start with no adjustment
    
    if sensors:
        if sensors.temperature_c and sensors.temperature_c > 5:  # Ideal fridge is 1-4°C
            # If temp is high, reduce shelf life
            adjustment_factor *= 0.75  # (e.g., 25% reduction)
            
        if sensors.humidity_percent and sensors.humidity_percent > 80:
            # If humidity is high, reduce shelf life for some items
            if label.lower() in ['lettuce', 'bread']:
                adjustment_factor *= 0.8  # (e.g., 20% reduction)

    adjusted_days = max(1, int(base_days * adjustment_factor)) # Ensure at least 1 day
    
    # Calculate the approximate expiration date
    approx_date = now + timedelta(days=adjusted_days)
    
    return f"{approx_date.isoformat()} (Approximation)"

def parse_vision_data(img: Image.Image, sensors: Optional[SensorData]) -> VisionData:
    """Runs YOLO and Tesseract on the image."""
    
    final_item_list = []
    full_ocr_text = ""
    
    if not model:
        # This will now be true if your custom model failed to load
        return VisionData(detected_items=[], raw_ocr_text="Error: YOLO model not loaded.")

    try:
        # 1. Run YOLO Object Detection
        results = model.predict(img, verbose=False, conf=0.5)
        
        if results and results[0].boxes:
            names = model.names # Use the model's names attribute
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes
            class_ids = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # 2. Run Zonal OCR: Run OCR on each item's bounding box
            for i in range(len(boxes)):
                label = names.get(int(class_ids[i]), "unknown")
                confidence = float(confs[i])
                
                # Crop the image to the bounding box
                box = boxes[i]
                cropped_img = img.crop((box[0], box[1], box[2], box[3]))
                
                # Run OCR on the cropped image
                item_ocr_text = pytesseract.image_to_string(cropped_img)
                full_ocr_text += f"\n--- Item: {label} ---\n{item_ocr_text}"
                
                # 3. Find date
                found_date = find_expiration_date(item_ocr_text)
                is_approx = False
                
                # 4. If no date found, approximate it
                if not found_date:
                    found_date = approximate_expiration(label, sensors)
                    is_approx = True
                
                final_item_list.append(
                    DetectedItem(
                        name=label,
                        expiration_date=found_date,
                        confidence=confidence,
                        is_approximate=is_approx
                    )
                )
        else:
            # No objects detected, run OCR on the whole image as a fallback
            full_ocr_text = pytesseract.image_to_string(img)

    except Exception as e:
        print(f"❌ Error during vision processing: {e}")
        return VisionData(detected_items=[], raw_ocr_text=f"Error: {e}")

    return VisionData(
        detected_items=final_item_list,
        raw_ocr_text=full_ocr_text.strip()
    )

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Python ML Service is running"}

@app.post("/process_image", response_model=VisionData)
def process_image(data: ImageData):
    """
    Receives a base66 encoded image and sensor data, runs ML models,
    and returns detected objects and text.
    """
    try:
        # Decode the base64 image
        img_data = base64.b64decode(data.image_base64)
        img = Image.open(io.BytesIO(img_data))
        
        # Run the vision processing pipeline
        vision_results = parse_vision_data(img, data.sensor_data)
        return vision_results

    except Exception as e:
        print(f"❌ Critical error in /process_image: {e}")
        # Return a valid VisionData object on error
        return VisionData(detected_items=[], raw_ocr_text=f"Error: {e}")