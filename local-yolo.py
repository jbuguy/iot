import sys
import json
import base64
import io
import os
import re
from PIL import Image
from ultralytics import YOLO
import pytesseract
from dateutil.parser import parse, ParserError

# Suppress STDOUT from ultralytics to keep our JSON clean
os.environ['ULTRALYTICS_LOGGING_LEVEL'] = 'ERROR'

# --- Model Loading ---
try:
    yolo_model = YOLO('yolov10n.pt')
except Exception as e:
    print(json.dumps({"error": f"Failed to load YOLO model: {e}"}), file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---

def decode_image(base64_string):
    """Converts a base64 string to a PIL Image."""
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception:
        return None

def run_object_detection(img):
    """Runs prediction and returns a list of unique object names."""
    try:
        results = yolo_model.predict(img, verbose=False, conf=0.5)
        detected_names = []
        if results and results[0].boxes:
            names = results[0].names
            for box in results[0].boxes:
                class_id = int(box.cls)
                label = names.get(class_id, "unknown")
                detected_names.append(label)
        return list(set(detected_names))
    except Exception as e:
        print(json.dumps({"error": f"Failed during YOLO prediction: {e}"}), file=sys.stderr)
        return []

def run_ocr(img):
    """Runs Tesseract OCR on the image and returns all found text."""
    try:
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(json.dumps({"error": f"Failed during Tesseract OCR: {e}"}), file=sys.stderr)
        return ""

def find_expiration_date(text_block):
    """
    Parses a block of text to find the most likely expiration date.
    Returns a date in 'YYYY-MM-DD' format or None.
    """
    # Regex patterns for common date formats
    # This can be expanded
    patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', # 12/31/2025, 12-31-25
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})', # 2025-12-31
        r'(\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})', # 31 Dec 2025
        r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{2,4})' # Dec 31, 2025
    ]
    
    # Simple keywords to look for
    keywords = ['exp', 'use by', 'best by', 'sell by']
    
    found_date = None
    
    # 1. Look for dates near keywords
    for line in text_block.lower().split('\n'):
        if any(key in line for key in keywords):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        # Use dateutil.parser to handle various formats
                        # 'dayfirst=False' assumes US-style MM/DD
                        parsed_date = parse(match.group(1), dayfirst=False, fuzzy=True)
                        return parsed_date.strftime('%Y-%m-%d')
                    except (ParserError, OverflowError):
                        continue # Ignore unparseable dates

    # 2. If no keyword match, look for any date in the text
    if not found_date:
        for pattern in patterns:
            match = re.search(pattern, text_block)
            if match:
                try:
                    parsed_date = parse(match.group(1), dayfirst=False, fuzzy=True)
                    # We only want dates in the future
                    if parsed_date.date() >= parsed_date.today().date():
                        return parsed_date.strftime('%Y-%m-%d')
                except (ParserError, OverflowError):
                    continue
                    
    return None

def main():
    try:
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input data received."}), file=sys.stderr)
            sys.exit(1)
            
        data = json.loads(input_data)
        
        if 'image_base64' not in data:
            print(json.dumps({"error": "No 'image_base64' field in input."}), file=sys.stderr)
            sys.exit(1)

        image = decode_image(data['image_base64'])
        if image is None:
            print(json.dumps({"error": "Failed to decode base64 image."}), file=sys.stderr)
            sys.exit(1)

        # --- Run ML Models ---
        detected_objects = run_object_detection(image)
        ocr_text = run_ocr(image)
        
        # --- Process Results ---
        # This is a simple logic: it finds ONE date and assigns it to ONE item.
        # A more advanced version would use object bounding boxes to crop and OCR each item.
        
        found_date = find_expiration_date(ocr_text)
        
        output_items = []
        
        # Heuristic: If we only find one object, assign the date to it.
        if len(detected_objects) == 1:
            output_items.append({
                "name": detected_objects[0],
                "expiration_date": found_date
            })
        elif len(detected_objects) > 0:
            # If multiple items, add them without dates, unless it's a date-specific item
            for item in detected_objects:
                # Simple heuristic: "milk" is a good candidate for the date
                if "milk" in item.lower() and found_date:
                     output_items.append({"name": item, "expiration_date": found_date})
                else:
                     output_items.append({"name": item, "expiration_date": None})
        
        # If no objects, but we found a date, just log it (logic can be improved)
        if not output_items and found_date:
             output_items.append({"name": "unknown_item", "expiration_date": found_date})

        print(json.dumps({"detected_items": output_items}))

    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input."}), file=sys.stderr)
    except Exception as e:
        print(json.dumps({"error": f"An unexpected main error occurred: {e}"}), file=sys.stderr)

if __name__ == "__main__":
    main()
    