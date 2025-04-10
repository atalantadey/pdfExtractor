# ruleEngine.py
import json
import csv
from tkinter import Image
import matplotlib.pyplot as plt
import logging
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Image as ReportLabImage, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import fitz  # PyMuPDF
import os
from PIL import Image as PILImage
from reportlab.pdfgen import canvas

from signal_v3 import perform_ocr, save_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname) - %(message)s')

# Load OCR Results
ocr_file = "output/ocr_results.json"
with open(ocr_file, "r") as f:
    ocr_data = json.load(f)

THRESHOLD = 0.5  # Adjust as needed
PDF_PATH = "input/sample3.pdf"
input_csv = "input/sample3.csv"
control_csv = "input/control_sample3.csv"
OUTPUT_COMBINED_DIR = "output/combined_error_pages"
OUTPUT_ERROR_PDF = "output/error_pages.pdf"
os.makedirs(OUTPUT_COMBINED_DIR, exist_ok=True)


def group_pages(ocr_data):
    """Groups pages based on strict page numbering ('Page 1 of N')."""
    grouped_sections = []
    current_group = []
    n_pages = len(ocr_data["pages"])
    i = 0

    while i < n_pages:
        page_info = ocr_data["pages"][i]
        page_number = page_info["page_number"]
        heading_preview = page_info["signals"]["heading"] if page_info["signals"]["heading"] else "No Heading"
        detected_dates = page_info["signals"]["date"]
        page_info_strict = page_info["signals"]["page_info_strict"]

        # Priority 1: Strict "Page 1 of N"
        if page_info_strict and page_info_strict[0] == 1 and page_info_strict[1] > 1:
            if current_group:
                grouped_sections.append(current_group)
                current_group = []  # Start a new group after the previous one

            start_page = page_number
            end_page = start_page + page_info_strict[1] - 1
            group = [(ocr_data["pages"][j]["page_number"],
                      ocr_data["pages"][j]["signals"]["heading"] if ocr_data["pages"][j]["signals"]["heading"] else "No Heading",
                      ocr_data["pages"][j]["signals"]["date"])
                     for j in range(i, min(end_page, n_pages))]
            grouped_sections.append(group)
            i = min(end_page, n_pages)
            continue  # Move to the next potential "Page 1 of N"

        # If no "Page 1 of N", simply add the current page to the current group
        current_group.append((page_number, heading_preview, detected_dates))
        i += 1

    # Add any remaining pages in the current group
    if current_group:
        grouped_sections.append(current_group)

    return grouped_sections


def save_to_csv(grouped_sections):
    """Save grouped sections into a CSV."""
    csv_file = "output/processed_subdocument_sample3.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for group in grouped_sections:
            if group:
                start_page = group[0][0]
                end_page = group[-1][0]
                heading = " | ".join(set(entry[1] for entry in group if entry[1] != "No Heading"))
                start_dates = [date for entry in group for date in (entry[2] if entry[2] else [])]
                end_dates = [date for entry in reversed(group) for date in (entry[2] if entry[2] else [])]
                start_date = start_dates[0] if start_dates else "No Date"
                end_date = end_dates[0] if end_dates else "No Date"
                writer.writerow([start_page, end_page, heading, start_date, end_date])

    print(f"✅ Processed CSV saved at {csv_file}")
    return csv_file


def load_input_csv(file_path, y_offset=0):
    """Load page groups from a CSV."""
    page_values = {}
    step_data = [(0, y_offset)]
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    start_page = int(row[0])
                    end_page = int(row[1])
                    for page in range(start_page, end_page + 1):
                        page_values[page] = 1
                    step_data.extend([(start_page, y_offset), (start_page, y_offset + 1), (end_page, y_offset + 1),
                                      (end_page, y_offset)])
                except ValueError as e:
                    logging.warning(f"Skipping invalid row in CSV: {row} - {e}")
    return step_data, page_values


def plot_comparison_chart(processed_file, input_file, control_file):
    """Compare processed grouping with input and control."""
    processed_data, processed_pages = load_input_csv(processed_file, y_offset=0)
    input_data, input_pages = load_input_csv(input_file, y_offset=4)
    control_data, control_pages = load_input_csv(control_file, y_offset=8)

    processed_segments = []
    for group in group_pages(ocr_data):
        if group:
            start_page = group[0][0]
            end_page = group[-1][0]
            processed_segments.append((start_page, end_page))

    filtered_processed_data = []
    for start, end in processed_segments:
        filtered_processed_data.extend([(start, 0), (start, 1), (end, 1), (end, 0)])

    plt.figure(figsize=(12, 8))
    plt.plot(*zip(*filtered_processed_data), marker='o', linestyle='-', color='b', label="Processed")
    plt.plot(*zip(*input_data), marker='o', linestyle='-', color='gray', label="Input")
    plt.plot(*zip(*control_data), marker='o', linestyle='-', color='g', label="Control")

    processed_all_pages = set()
    for start, end in processed_segments:
        processed_all_pages.update(range(start, end + 1))

    error_input = {p for p in processed_all_pages if p not in input_pages} | {p for p in input_pages if
                                                                               p not in processed_all_pages}
    error_control = {p for p in processed_all_pages if p not in control_pages} | {p for p in control_pages if
                                                                                   p not in processed_all_pages}

    if error_input:
        plt.scatter(list(error_input), [4] * len(error_input), color='r', marker='x', s=100, label="Input Errors")
    if error_control:
        plt.scatter(list(error_control), [8] * len(error_control), color='orange', marker='x', s=100,
                    label="Control Errors")

    plt.xlabel("Page Numbers")
    plt.ylabel("Grouping")
    plt.title("Comparison of Page Grouping")
    plt.yticks([0, 1, 4, 5, 8, 9], ["False (P)", "True (P)", "False (I)", "True (I)", "False (C)", "True (C)"])
    plt.grid(True)
    plt.legend()
    plt.savefig("output/group_comparison_chart.png")
    plt.show()
    print("📊 Graph saved at output/group_comparison_chart.png")
    return error_input, error_control


def calculate_error_percentage(processed_file, input_file, control_file):
    """Calculate error percentage."""

    def extract_pages(file_path):
        pages = set()
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        start_page, end_page = map(int, row[:2])
                        pages.update(range(start_page, end_page + 1))
                    except ValueError as e:
                        logging.warning(f"Invalid row in CSV: {row} - {e}")
        return pages

    processed_pages = set()
    for group in group_pages(ocr_data):
        if group:
            start_page = group[0][0]
            end_page = group[-1][0]
            processed_pages.update(range(start_page, end_page + 1))

    input_pages = extract_pages(input_file)
    control_pages = extract_pages(control_file)

    total_relevant_pages = processed_pages.union(input_pages).union(control_pages)
    error_input = len(processed_pages.symmetric_difference(input_pages))
    error_control = len(processed_pages.symmetric_difference(control_pages))

    percentage_input = (error_input / len(total_relevant_pages)) * 100 if total_relevant_pages else 0
    percentage_control = (error_control / len(total_relevant_pages)) * 100 if total_relevant_pages else 0

    print(f"\n✅ Error Percentage (Input): {percentage_input:.2f}%")
    print(f"\n✅ Error Percentage (Control): {percentage_control:.2f}%\n")
    return percentage_input, percentage_control, control_pages, processed_pages, input_pages, control_pages


def get_page_image(doc, page_num, img_path):
    """Extracts and saves a page as an image."""
    try:
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
            pix.save(img_path)
            return True
        return False
    except Exception as e:
        logging.error(f"Error extracting image for page {page_num}: {e}")
        return False


def create_error_pdf(error_pages, pdf_path, output_pdf_path):
    """Creates a PDF containing the combined error pages with headings."""
    doc = fitz.open(pdf_path)
    pdf_writer = canvas.Canvas(output_pdf_path, pagesize=letter)

    num_pages = len(doc)
    padding = 10

    for error_page_num in sorted(list(error_pages)):
        pil_images = []
        widths = []
        max_height = 0
        temp_files_to_remove = []

        for i in range(max(1, error_page_num - 1), min(num_pages + 1, error_page_num + 2)):
            temp_path = os.path.join(OUTPUT_COMBINED_DIR, f"temp_page_{error_page_num}_{i}.png")
            if get_page_image(doc, i, temp_path):
                try:
                    img = PILImage.open(temp_path)
                    pil_images.append(img)
                    widths.append(img.width)
                    max_height = max(max_height, img.height)
                    temp_files_to_remove.append(temp_path)
                except FileNotFoundError:
                    logging.error(f"Could not open image: {temp_path}")
            else:
                blank_img = PILImage.new('RGB', (100, max_height if max_height > 0 else 100),
                                         color='white')  # Placeholder
                pil_images.append(blank_img)
                widths.append(blank_img.width)

        if pil_images:
            total_width = sum(widths) + padding * (len(pil_images) - 1)
            combined_image = PILImage.new('RGB', (total_width, max_height), color='white')
            x_offset = 0
            for img in pil_images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.width + padding

            output_combined_path = os.path.join(OUTPUT_COMBINED_DIR,
                                                f"combined_error_page_{error_page_num}.png")
            combined_image.save(output_combined_path)
            print(
                f"✅ Combined image saved for error page {error_page_num} at {output_combined_path}")  # Debugging print

            img_width_inch = min(letter[0] - 2 * inch, total_width / 72)  # Max width to fit page
            img_height_inch = (img_width_inch / total_width) * max_height / 72
            y_position = letter[1] - inch - img_height_inch

            # Remove heading
            pdf_writer.drawImage(output_combined_path, inch, y_position, width=img_width_inch,
                                height=img_height_inch)
            pdf_writer.showPage()

            for img in pil_images:
                img.close()

        for temp_file in temp_files_to_remove:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    pdf_writer.save()
    doc.close()
    print(f"✅ Error pages combined into PDF: {output_pdf_path}")


if __name__ == "__main__":
    ocr_results = None
    if os.path.exists(ocr_file):
        with open(ocr_file, "r") as f:
            ocr_results = json.load(f)
    else:
        doc = fitz.open(PDF_PATH)
        images = [Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height],
                                    page.get_pixmap().samples) for page in doc]
        ocr_results = perform_ocr(images)
        save_results(ocr_results)
        doc.close()
        print(f"OCR processing completed and saved to {ocr_file}")

    if ocr_results:
        grouped_sections = group_pages(ocr_results)
        csv_file = save_to_csv(grouped_sections)

        error_input, error_control = plot_comparison_chart(csv_file, input_csv, control_csv)
        error_percentage_input, error_percentage_control, control_pages, processed_pages, input_pages, control_pages = calculate_error_percentage(
            csv_file, input_csv, control_csv)

        print("\n🚨 Pages with Errors (Input):", sorted(error_input))
        print("\n🚨 Pages with Errors (Control):", sorted(error_control))
        print(f"✅ Combined error page images saved in: {OUTPUT_COMBINED_DIR}")
        create_error_pdf(error_control, PDF_PATH, OUTPUT_ERROR_PDF)
        print(f"✅ Combined error pages PDF saved at {OUTPUT_ERROR_PDF}")
        # create_error_pdf(error_input, PDF_PATH, OUTPUT_ERROR_PDF.replace(".pdf", "_input_errors.pdf"))