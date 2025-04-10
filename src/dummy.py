import json
import os
from PIL import Image
import fitz # PyMuPDF
import spacy  # Import spaCy for NLP-based matching


output_dir="output/"
sample="1"
# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")  # Using a larger model
except OSError:
    print("Downloading en_core_web_md model for spaCy...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    
def perform_ocr(images):
    return []
    
    
def main():
    ocr_file=readfile() 
    print(f"OCR processing completed and saved to {ocr_file}")
    
    
    
    
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
    
