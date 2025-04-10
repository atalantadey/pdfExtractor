import json
import logging
import os
import re
from PIL import Image,ImageEnhance, ImageFilter
import fitz # PyMuPDF
import spacy  # Import spaCy for NLP-based matching
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


sample="5"
OCR_CONFIG = "--psm 1 --oem 3"  # PSM 1 for automatic with OSD, OEM 3 for best engine
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
pdf_path = "input/sample1.pdf"
ocr_path = os.path.join(output_dir, f"ocr_results{sample}.json")
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")  # Using a larger model
except OSError:
    print("Downloading en_core_web_md model for spaCy...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    
def extract_page_number_strict(text):
    """Extract page number in 'Page X of Y' format only."""
    pattern = r'(page|PAGE)\s+(\d+)\s+of\s+(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            current_page = int(match.group(2))
            total_pages = int(match.group(3))
            if current_page > 0 and total_pages > 0 and current_page <= 20 and total_pages <= 20 and current_page <= total_pages:
                return current_page, total_pages
            else:
                return None
        except ValueError:
            return None
    return None

def extract_page_number_general(text):
    """Extract page number in various formats (including 'pg X of Y')."""
    patterns = [
        r'(page|PAGE)\s+(\d+)\s+of\s+(\d+)',
        r'pg\s+(\d+)\s+of\s+(\d+)',
        r'(\d+)\s+of\s+(\d+)',
        r'-?\s*(\d+)\s*-?',
        r'\((\d+)\s+of\s+\d+\)',
        r'(\d+)\/\d+',
        r'(?:seite|s\.)\s+(\d+)',
        r'(\d+)'  # Catch single numbers (less reliable, use with caution)
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and len(match.groups()) >= 1 and match.group(1).isdigit():
            try:
                current_page_general = int(match.group(1))
                total_page_general = int(match.group(2)) if len(match.groups()) > 1 and match.group(2) and match.group(2).isdigit() else None
                if total_page_general is not None:
                    if current_page_general > 0 and total_page_general > 0 and current_page_general <= 20 and total_page_general <= 20 and current_page_general <= total_page_general:
                        return current_page_general, total_page_general
                    else:
                        continue # Skip if benchmark not met
                elif len(match.groups()) == 1 and current_page_general > 0 and current_page_general <= 20:
                    return current_page_general, None # Single page number
                elif len(match.groups()) >= 2 and total_page_general is None and current_page_general > 0 and current_page_general <= 20:
                    return current_page_general, None
                elif len(match.groups()) >= 2 and total_page_general is not None and current_page_general > 0 and total_page_general > 0 and current_page_general <= 20 and total_page_general <= 20 and current_page_general <= total_page_general:
                    return current_page_general, total_page_general
                elif len(match.groups()) == 1 and current_page_general > 0 and current_page_general <= 20:
                    return current_page_general, None

            except ValueError:
                continue
    return None

def extract_date(text):
    """Extract dates with various formats using regex and NLP."""
    dates = set()
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:of\s+)?\d{4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(?:of\s+)?\d{4}\b',
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{8}\b',
        r'\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2,4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}-\d{2,4}\b',
    ]
    for pattern in date_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            dates.update(found)

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)

    return list(dates) if dates else None

def extract_heading(text):
    """Extract the first non-empty line as a potential heading."""
    lines = text.split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            return cleaned_line[:350]
    return None

def longest_common_substring(s1, s2):
    """Find the longest common substring and its length."""
    if not s1 or not s2:
        return ""
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = "", 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1].lower() == s2[y - 1].lower():
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > len(longest):
                    longest = s1[x - m[x][y]:x]
    return longest

def perform_ocr_accurate(image):
    """Perform accurate OCR with multiple processing steps."""
    text = ""
    # 1. Basic OCR
    text += pytesseract.image_to_string(image, config=OCR_CONFIG) + "\n"

    # 2. Grayscale and slight contrast
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.2)
    text += pytesseract.image_to_string(enhanced, config=OCR_CONFIG) + "\n"

    # 3. Binarization (adaptive thresholding might be better for complex backgrounds)
    try:
        thresh = gray.point(lambda p: 255 if p > 150 else 0, mode='1')
        text += pytesseract.image_to_string(thresh, config=OCR_CONFIG) + "\n"
    except Exception as e:
        logging.warning(f"Error during binarization: {e}")

    # 4. Noise reduction (might need more sophisticated techniques)
    blurred = gray.filter(ImageFilter.MedianFilter(3))
    text += pytesseract.image_to_string(blurred, config=OCR_CONFIG) + "\n"

    return text.strip()

def process_page(page_number, image, ):
    """Perform OCR and extract key signals for a page."""
    signals = {
        "page_info_strict": None,
        "page_info_general": None,
        "date": None,
        "heading": None,
        "full_text": ""
    }
    try:
        signals["full_text"] = perform_ocr_accurate(image)
        print(f"OCR successful for page {page_number}")
    except Exception as e:
        print(f"Error during OCR for page {page_number}: {e}")
        
    signals["page_info_strict"] = extract_page_number_strict(signals["full_text"])
    signals["page_info_general"] = extract_page_number_general(signals["full_text"])
    signals["date"] = extract_date(signals["full_text"])
    signals["heading"] = extract_heading(signals["full_text"])

    weight = 0
    continuity_reasons = {}    
    
    return {
        "page_number": page_number,
        "signals": signals,
        "continuity_weight": weight,
        "continuity_reasons": continuity_reasons,
        "full_text_preview": signals["full_text"][:350] if signals["full_text"] else None,
    },signals

def perform_ocr(images):
    """Perform OCR on all pages and process results."""
    results = {"pages": [], "continuity_weights": []}
    page_number_counts = {}
    date_counts = {}
    strict_page_numbers = {}
    general_page_numbers = {}

    for i, img in enumerate(images):
        page_result, current_signals = process_page(i+1, img)
        results["pages"].append(page_result)
    
    # Store counts for analysis
        num_dates = len(current_signals["date"]) if current_signals["date"] else 0
        date_counts[i + 1] = num_dates

        page_info_strict = current_signals["page_info_strict"]
        if page_info_strict:
            strict_page_numbers[i + 1] = page_info_strict

        page_info_general = current_signals["page_info_general"]
        if page_info_general:
            general_page_numbers[i + 1] = page_info_general

        num_page_info = 1 if page_info_strict else 0
        page_number_counts[i + 1] = num_page_info

    # Print analysis
    total_dates = sum(date_counts.values())
    total_strict_page_numbers = len(strict_page_numbers)
    total_general_page_numbers = len(general_page_numbers)
    print("\n--- OCR Analysis ---")
    print(f"Total Dates Extracted: {total_dates}")
    print(f"Total 'Page X of Y' Formats Found: {total_strict_page_numbers}")
    if strict_page_numbers:
        print("Page Numbers ('Page X of Y') and their Locations:")
        for page, numbers in strict_page_numbers.items():
            print(f"  Page {page}: Current - {numbers[0]}, Total - {numbers[1]}")
    print("----------------------\n")

    return results
       
def main():
    ocr_file=readfile() 
    print(f"OCR processing completed and saved to {ocr_path}")
        
def readfile():
    pdf_path = f"input/sample{sample}.pdf"
    doc = fitz.open(pdf_path)
    images = [Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples) for page in doc]
    ocr_results = perform_ocr(images)
    save_results(ocr_results)
    return pdf_path
    
    
def save_results(results):
    """Save results to JSON."""
    json_path = os.path.join(output_dir, f"ocr_results{sample}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
    
