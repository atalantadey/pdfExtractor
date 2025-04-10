# SignalExtraction.py
import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from difflib import SequenceMatcher
import logging
import spacy  # Import spaCy for NLP-based matching
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurations
OCR_CONFIG = "--psm 1 --oem 3"  # PSM 1 for automatic with OSD, OEM 3 for best engine
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
pdf_path = "input/sample5.pdf"
ocr_file = os.path.join(output_dir, "ocr_results.json")

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
            return current_page, total_pages
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
        if match:
            try:
                page_num = int(match.group(1))
                total_num = int(match.group(2)) if len(match.groups()) > 1 and match.group(2) else None
                return page_num, total_num
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


def process_page(page_number, image, prev_signals):
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

    if prev_signals:
        # Strongest signal: Consecutive 'Page X of Y' with same total
        if prev_signals.get("page_info_strict") and signals["page_info_strict"]:
            prev_pn, prev_total = prev_signals["page_info_strict"]
            curr_pn, curr_total = signals["page_info_strict"]
            if prev_total is not None and curr_total is not None and prev_total == curr_total and prev_pn + 1 == curr_pn:
                weight = 0.9  # Very strong continuity
                continuity_reasons["strong_page_number_continuity_strict"] = True
            elif curr_pn == 1 and curr_total is not None and curr_total > 1:
                weight = 0.95  # Even stronger for start of a new sub-document
                continuity_reasons["new_subdocument_page_1_strict"] = True

        # General Page Number Continuity (including 'pg X of Y')
        if not continuity_reasons.get("strong_page_number_continuity_strict") and prev_signals.get("page_info_general") and signals["page_info_general"]:
            prev_pn_gen, prev_total_gen = prev_signals["page_info_general"]
            curr_pn_gen, curr_total_gen = signals["page_info_general"]
            if prev_pn_gen is not None and curr_pn_gen is not None and prev_pn_gen + 1 == curr_pn_gen and (prev_total_gen is None or curr_total_gen is None or prev_total_gen == curr_total_gen):
                weight = max(weight, 0.7)
                continuity_reasons["general_page_number_continuity"] = True

        # Longest Common Substring for Headings
        if prev_signals.get("heading") and signals["heading"]:
            lcs = longest_common_substring(prev_signals["heading"], signals["heading"])
            if len(lcs) > 0.8 * min(len(prev_signals["heading"]), len(signals["heading"])):
                weight = max(weight, 0.6)
                continuity_reasons["heading_similarity_lcs"] = True
            else:
                # Fallback to NLP similarity if LCS is not strong
                similarity = nlp(prev_signals["heading"]).similarity(nlp(signals["heading"]))
                if similarity > 0.85:
                    weight = max(weight, 0.5)
                    continuity_reasons["heading_similarity_nlp"] = True

        # Date Proximity (still relevant)
        if prev_signals.get("date") and signals["date"]:
            for d1_str in prev_signals["date"]:
                for d2_str in signals["date"]:
                    try:
                        # Attempt to parse dates (more robust)
                        date_format = None
                        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y%m%d', '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y', '%m-%d-%Y', '%d-%b-%Y', '%b-%d-%Y'):
                            try:
                                date1 = datetime.strptime(d1_str.strip(), fmt)
                                date2 = datetime.strptime(d2_str.strip(), fmt)
                                date_format = fmt
                                break
                            except ValueError:
                                pass

                        if date_format:
                            if abs((date2 - date1).days) <= 32:
                                weight = max(weight, 0.3)
                                continuity_reasons["date_proximity"] = True
                                break  # Move to the next date pair
                    except ValueError:
                        pass  # Ignore unparseable dates
                if "date_proximity" in continuity_reasons:
                    break

    return {
        "page_number": page_number,
        "signals": signals,
        "continuity_weight": round(weight, 5),
        "continuity_reasons": continuity_reasons,
        "full_text_preview": signals["full_text"][:200]
    }, signals


def perform_ocr(images):
    """Perform OCR on all pages and process results."""
    results = {"pages": [], "continuity_weights": []}
    prev_signals = {}
    page_number_counts = {}
    date_counts = {}
    strict_page_numbers = {}
    general_page_numbers = {}

    for i, img in enumerate(images):
        page_result, current_signals = process_page(i + 1, img, prev_signals)
        results["pages"].append(page_result)
        prev_signals = current_signals

        # Reset continuity weight based on strong page number continuity
        if i > 0 and not results["pages"][i]["continuity_reasons"].get("strong_page_number_continuity_strict") and not results["pages"][i]["continuity_reasons"].get("new_subdocument_page_1_strict"):
            results["continuity_weights"].append(0.0)
        elif i > 0:
            results["continuity_weights"].append(results["continuity_weights"][-1] + results["pages"][i]["continuity_weight"])
        else:
            results["continuity_weights"].append(results["pages"][i]["continuity_weight"])

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
    if general_page_numbers:
        print("\nGeneral Page Numbers (including 'pg X of Y') and their Locations:")
        for page, numbers in general_page_numbers.items():
            print(f"  Page {page}: Current - {numbers[0]}, Total - {numbers[1] if numbers[1] else 'N/A'}")
    if date_counts:
        print("\nNumber of Dates Found per Page:")
        for page, count in date_counts.items():
            print(f"  Page {page}: {count}")
    print("----------------------\n")

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
    print(f"OCR processing completed and saved to {ocr_file}")