from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import os

from pdf2image import convert_from_bytes
import google.generativeai as genai
from typing import List
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Set Tesseract path (adjust if needed)
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract"
]

for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break
else:
    print("Warning: Tesseract not found in common locations. Please install Tesseract OCR.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5173", "http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY environment variable not set. Please check your .env file or environment variables.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def clean_ocr_text(text):
    """
    Clean and preprocess OCR text to improve accuracy
    """
    if not text:
        return text
    
    import re
    
    # Fix common OCR errors
    replacements = {
        'Lable': 'Label',
        'Chola': 'Chola',
        '+|bCAmt:+/2¢': 'SubTotal:',
        'Phn:': 'Phone:',
        'Qty:': 'Quantity:',
        'Amt:': 'Amount:',
        'Cewkehy': 'Company',
        'Srey': 'Store'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean up excessive whitespace but preserve bill dividers
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Ensure proper spacing around numbers and currency
    text = re.sub(r'(\d+)\.(\d+)', r'\1.\2', text)
    
    return text.strip()

@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    print(f"Received request with {len(files) if files else 0} files")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    all_text = ""
    bill_counter = 1
    
    for file in files:
        print(f"Processing file: {file.filename}, type: {file.content_type}")
        if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
        
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        # Add clear bill header divider
        all_text += f"\n\n{'='*60}\n"
        all_text += f"BILL NUMBER {bill_counter} - START\n"
        all_text += f"{'='*60}\n\n"
        
        try:
            if file.content_type == "application/pdf":
                print("Converting PDF to images...")
                # Convert PDF to images
                images = convert_from_bytes(contents)
                print(f"PDF has {len(images)} pages")
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(image)
                    print(f"Page {i+1} text length: {len(text)}")
                    all_text += text + "\n"
            else:
                print("Processing image...")
                # Process image with enhanced OCR settings
                image = Image.open(io.BytesIO(contents))
                
                # Try multiple OCR configurations for better accuracy
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,/-:()+'
                
                # First attempt with custom config
                text1 = pytesseract.image_to_string(image, config=custom_config)
                
                # Second attempt with different PSM mode for line detection
                config2 = r'--oem 3 --psm 4'
                text2 = pytesseract.image_to_string(image, config=config2)
                
                # Third attempt with automatic page segmentation
                config3 = r'--oem 3 --psm 3'
                text3 = pytesseract.image_to_string(image, config=config3)
                
                # Choose the longest/most complete text
                texts = [text1, text2, text3]
                text = max(texts, key=len)
                
                print(f"Image text length: {len(text)}")
                print(f"Extracted text preview: {text[:300]}...")
                all_text += text + "\n"
                
            # Add clear bill footer divider
            all_text += f"\n{'='*60}\n"
            all_text += f"BILL NUMBER {bill_counter} - END\n"
            all_text += f"{'='*60}\n\n"
            bill_counter += 1
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
    
    print(f"Total extracted text length: {len(all_text)}")
    if not all_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from images")
    
    # Clean up the OCR text
    cleaned_text = clean_ocr_text(all_text)
    
    print("Sending to Gemini...")
    # Send to Gemini
    print("Cleaned text:")
    print(cleaned_text)
    prompt_template = """
You are an intelligent document processor that extracts bill/invoice data from OCR text into a unified table.

The text contains MULTIPLE DIFFERENT BILLS/INVOICES separated by clear dividers.

BILL SEPARATION MARKERS:
Each bill is clearly marked with:
- "BILL NUMBER X - START" at the beginning
- "BILL NUMBER X - END" at the end
- "=" dividers around each bill
- "SOURCE FILE: filename" to identify the original image

CRITICAL INSTRUCTIONS FOR MULTIPLE BILLS:
1. PROCESS each bill section separately (between START and END markers)
2. Each bill has its own date - FIND AND USE the correct date for each bill's items
3. Each bill may have different shop names - apply the correct shop name to its items
4. DO NOT mix dates between different bills
5. Process each bill individually, then combine all items into one JSON array

IMPORTANT DATE HANDLING:
- IDENTIFY each bill's individual date within its START/END section
- Look for dates like: 27-Jan-2012, 12/03/2024, 15-Mar-2024, 12:18 PM 27-Jan-2012
- Convert each date to DD/MM/YYYY format (example: "27-Jan-2012" becomes "27/01/2012")
- Apply the CORRECT date to ALL items from that specific bill only
- Handle various date formats: 02/03/2024, 02-March-2024, 2024-03-02, 02 Mar 2024, etc.
- If a bill has no clear date, use current date (16/10/2025)

BILL PROCESSING EXAMPLE:
If you see:
```
=== BILL NUMBER 1 - START ===
... 27-Jan-2012 ... Restaurant A ... Rice, Oil ...
=== BILL NUMBER 1 - END ===

=== BILL NUMBER 2 - START ===  
... 15-Mar-2024 ... Grocery B ... Milk, Bread ...
=== BILL NUMBER 2 - END ===
```

Process as:
- Bill 1 items get date "27/01/2012" and shop "Restaurant A"
- Bill 2 items get date "15/03/2024" and shop "Grocery B"

PROCESSING STEPS:
1. Split the text into separate bills
2. For each bill: extract date, shop name, and all items
3. Apply the bill's date and shop name to all its items
4. Combine all items from all bills into one JSON array

Example with multiple bills:
If you find:
- Bill 1 (Date: 27/01/2012, Shop: Restaurant A) with items: Rice, Oil
- Bill 2 (Date: 15/03/2024, Shop: Grocery B) with items: Milk, Bread
- Bill 3 (Date: 20/05/2024, Shop: Store C) with items: Tea, Sugar

Output should be:
[
  {"date": "27/01/2012", "shop_name": "Restaurant A", "item_name": "Rice", "quantity": 1, "unit_price": 50, "total_amount": 50},
  {"date": "27/01/2012", "shop_name": "Restaurant A", "item_name": "Oil", "quantity": 1, "unit_price": 150, "total_amount": 150},
  {"date": "15/03/2024", "shop_name": "Grocery B", "item_name": "Milk", "quantity": 2, "unit_price": 40, "total_amount": 80},
  {"date": "15/03/2024", "shop_name": "Grocery B", "item_name": "Bread", "quantity": 1, "unit_price": 25, "total_amount": 25},
  {"date": "20/05/2024", "shop_name": "Store C", "item_name": "Tea", "quantity": 1, "unit_price": 30, "total_amount": 30},
  {"date": "20/05/2024", "shop_name": "Store C", "item_name": "Sugar", "quantity": 1, "unit_price": 45, "total_amount": 45}
]

Rules:
- Use field names: date, shop_name, item_name, quantity, unit_price, total_amount
- EVERY item must have the correct date from its specific bill
- Clean up item names and fix obvious OCR errors
- Calculate missing quantities or totals if needed
- Return only valid JSON array, no explanations

Text to process:
__ALL_TEXT_PLACEHOLDER__
"""

    prompt = prompt_template.replace("__ALL_TEXT_PLACEHOLDER__", cleaned_text)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        print(f"Gemini response length: {len(raw_text)}")
        print(f"Gemini response preview: {raw_text[:200]}...")
        
        # Clean the response - remove any markdown code blocks
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        structured_data = json.loads(raw_text)
        print("Successfully parsed JSON from Gemini")
        return {"data": structured_data}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print(f"Raw text: {raw_text}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON from Gemini: {raw_text[:200]}... Error: {str(e)}")
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Image Text Extraction API"}
