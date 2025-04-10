import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
from difflib import SequenceMatcher

# Configurations
OCR_CONFIG = "--psm 6 --oem 1"
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
pdf_path = "input/sample5.pdf"

WEIGHTS = {
    "heading_match": 0.5,
    "date_match": 0.3,
    "footer_match": 0.2
}

THRESHOLD = 0.3  # Threshold for determining continuity

def extract_text_top(text, num_lines=5):
    """Extract the first few lines from text (used for heading detection)."""
    lines = text.split("\n")[:num_lines]
    return " ".join(lines).strip()

def extract_date(text):
    """Extract dates using regex."""
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Formats like 12/03/2024 or 12-03-24
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'  # Formats like March 27, 2025
    ]
    matches = []
    for pattern in date_patterns:
        matches.extend(re.findall(pattern, text, re.IGNORECASE))
    return matches if matches else []

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

    top_text = extract_text_top(full_text)
    heading = top_text.split("\n")[0] if "\n" in top_text else top_text
    detected_dates = extract_date(full_text)
    
    # Determine continuity weight
    weight = 0
    if prev_signals:
        if prev_signals["heading"] and heading:
            lcs = longest_common_substring(prev_signals["heading"], heading)
            if len(lcs) > 3:
                weight += WEIGHTS["heading_match"]
        if prev_signals["dates"] and detected_dates:
            if any(date in prev_signals["dates"] for date in detected_dates):
                weight += WEIGHTS["date_match"]

    return {
        "page_number": page_number,
        "signals": {
            "heading": heading,
            "dates": detected_dates
        },
        "continuity_weight": round(weight, 5),
        "full_text": full_text[:200]  # Storing first 200 chars for debugging
    }, {
        "heading": heading,
        "dates": detected_dates
    }

def perform_ocr(images):
    """Perform OCR on all pages and process results."""
    results = {"pages": [], "cumulative_weights": []}
    prev_signals = {"heading": "", "dates": []}
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
