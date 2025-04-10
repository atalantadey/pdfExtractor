import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
from difflib import SequenceMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurations
OCR_CONFIG = "--psm 6 --oem 1"
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
pdf_path = "input/sample5.pdf"

WEIGHTS = {
    "heading_match": 0.4,
    "date_match": 0.3,
    "sheet_match": 0.3
}

THRESHOLD = 0.3  # Threshold for determining continuity

def extract_date_regex(text):
    """Extract dates using regex."""
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Formats like 12/03/2024 or 12-03-24
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'  # Formats like March 27, 2025
    ]
    matches = []
    for pattern in date_patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    return matches if matches else None

def extract_sheet_info_regex(text):
    """Extract sheet number and total sheets using regex (Page X of Y)."""
    sheet_match = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', text, re.IGNORECASE)
    if sheet_match:
        return int(sheet_match.group(1)), int(sheet_match.group(2))
    return None

def longest_common_substring(s1, s2):
    """Find the longest common substring between two texts."""
    seq_match = SequenceMatcher(None, s1, s2)
    match = seq_match.find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size] if match.size >= 3 else ""

def process_page(page_number, image, prev_signals):
    """Perform OCR and extract key signals for a page."""
    try:
        full_text = pytesseract.image_to_string(image, config=OCR_CONFIG).strip()
        print(f"OCR successful for page {page_number}")
    except Exception as e:
        print(f"Error during OCR for page {page_number}: {e}")
        full_text = ""

    # Extract signals from full text
    heading = full_text.split("\n")[0].strip() if full_text else None #first line
    detected_dates = extract_date_regex(full_text)
    sheet_info = extract_sheet_info_regex(full_text)

    # Determine continuity weight
    weight = 0
    current_page_signals = {}

    if prev_signals:
        # Heading Match
        if prev_signals.get("heading") and heading:
            lcs = longest_common_substring(prev_signals["heading"], heading)
            if len(lcs) > 3:
                weight += WEIGHTS["heading_match"]
                current_page_signals["heading_match"] = True
            else:
                current_page_signals["heading_match"] = False
        else:
            current_page_signals["heading_match"] = False

        # Date Match (checking for any match in the lists)
        date_match = False
        if prev_signals.get("dates") and detected_dates:
            for prev_date in prev_signals["dates"]:
                for current_date in detected_dates:
                    if prev_date == current_date:
                        weight += WEIGHTS["date_match"]
                        date_match = True
                        break
                if date_match:
                    break
        current_page_signals["date_match"] = date_match

        # Sheet Continuity
        sheet_continuity = False
        if prev_signals.get("sheet_info") and sheet_info and prev_signals["sheet_info"][1] == sheet_info[1] and prev_signals["sheet_info"][0] + 1 == sheet_info[0]:
            weight += WEIGHTS["sheet_match"]
            sheet_continuity = True
        current_page_signals["sheet_continuity"] = sheet_continuity

    return {
        "page_number": page_number,
        "signals": {
            "heading": heading,
            "dates": detected_dates,
            "sheet_info": sheet_info
        },
        "continuity_weight": round(weight, 5),
        "full_text": full_text[:200]  # Storing first 200 chars for debugging
    }, {
        "heading": heading,
        "dates": detected_dates,
        "sheet_info": sheet_info
    }

def perform_ocr(images):
    """Perform OCR on all pages and process results."""
    results = {"pages": [], "cumulative_weights": []}
    prev_signals = {"heading": None, "dates": None, "sheet_info": None}
    cumulative_weight = 0

    for i, img in enumerate(images):
        page_result, prev_signals = process_page(i + 1, img, prev_signals)
        cumulative_weight = round(cumulative_weight + page_result["continuity_weight"], 5) if page_result["continuity_weight"] > 0 else 0
        results["pages"].append(page_result)
        results["cumulative_weights"].append(cumulative_weight)

    return results

def save_results(results):
    """Save results to JSON."""
    json_path = os.path.join(output_dir, "ocr_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    doc = fitz.open(pdf_path)
    images = [Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples) for page in doc]
    ocr_results = perform_ocr(images)
    save_results(ocr_results)
    print(f"OCR processing completed. Cumulative Weights: {ocr_results['cumulative_weights']}")