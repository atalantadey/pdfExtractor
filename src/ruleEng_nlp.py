import json
import csv
import matplotlib.pyplot as plt

# Load OCR Results
ocr_file = "output/ocr_results.json"
with open(ocr_file, "r") as f:
    ocr_data = json.load(f)

def group_pages(ocr_data):
    """Groups pages based on continuity weights & dates."""
    grouped_sections = []
    current_group = []

    for i, page in enumerate(ocr_data["pages"]):
        page_number = page["page_number"]
        heading_preview = page["signals"]["heading"] if page["signals"]["heading"] else "No Heading"
        detected_dates = page["signals"]["dates"]
        continuity_weight = ocr_data["cumulative_weights"][i]

        if continuity_weight > 0:
            current_group.append((page_number, heading_preview, detected_dates))
        else:
            if len(current_group) > 1:
                grouped_sections.append(current_group)
            current_group = [(page_number, heading_preview, detected_dates)]
    
    if len(current_group) > 1:
        grouped_sections.append(current_group)

    return grouped_sections

def save_to_csv(grouped_sections):
    """Save grouped sections into a CSV without column headers in the first row."""
    csv_file = "output/processed_subdocument_sample3.csv"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        
        for group in grouped_sections:
            start_page = min(p[0] for p in group)
            end_page = max(p[0] for p in group)
            heading = " | ".join(set(entry[1] for entry in group if entry[1] != "No Heading"))
            start_date = next((date for entry in group for date in entry[2]), "No Date")
            end_date = next((date for entry in reversed(group) for date in entry[2]), "No Date")
            writer.writerow([start_page, end_page, heading, start_date, end_date])

    print(f"✅ Processed CSV saved at {csv_file}")
    return csv_file

def load_input_csv(file_path, y_offset=0):
    """Load page groups from a CSV file and return step-wise list for plotting with optional Y offset."""
    page_values = {}  # Store (page_number: 1/0) values for XOR operation
    step_data = [(0, y_offset)]  # Start at (0, y_offset)

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        for row in rows:
            start_page = int(row[0])  
            end_page = int(row[1])

            for page in range(start_page, end_page + 1):
                page_values[page] = 1  # Mark these pages as grouped (1)

            # Step-wise movement for visualization
            step_data.append((start_page, y_offset))
            step_data.append((start_page, y_offset + 1))
            step_data.append((end_page, y_offset + 1))
            step_data.append((end_page, y_offset))
    
    return step_data, page_values

def plot_comparison_chart(processed_file, input_file, control_file):
    """Compare processed CSV with both input and control CSVs using step-line charts and XOR-based error detection."""

    processed_data, processed_pages = load_input_csv(processed_file, y_offset=0)  # Processed CSV (Y=0)
    input_data, input_pages = load_input_csv(input_file, y_offset=4)  # Input CSV (Y=4)
    control_data, control_pages = load_input_csv(control_file, y_offset=8)  # Control CSV (Y=8)

    # Extract step-wise values for visualization
    processed_pages_x, processed_y = zip(*processed_data)
    input_pages_x, input_y = zip(*input_data)
    control_pages_x, control_y = zip(*control_data)

    # XOR Errors
    error_pages_input = {page for page in (processed_pages.keys() | input_pages.keys()) if processed_pages.get(page, 0) ^ input_pages.get(page, 0)}
    error_pages_control = {page for page in (processed_pages.keys() | control_pages.keys()) if processed_pages.get(page, 0) ^ control_pages.get(page, 0)}

    # Plot Step Line Graph
    plt.figure(figsize=(12, 8))
    
    # Processed Data (Blue)
    plt.plot(processed_pages_x, processed_y, marker='o', linestyle='-', color='b', label="Processed Groups (Y=0)")

    # Input Data (Gray, Y=4)
    plt.plot(input_pages_x, input_y, marker='o', linestyle='-', color='gray', label="Input Groups (Y=4)")

    # Control CSV Data (Green, Y=8)
    plt.plot(control_pages_x, control_y, marker='o', linestyle='-', color='g', label="Control Groups (Y=8)")

    # Error Markers for Input Comparison (Red 'X')
    if error_pages_input:
        plt.scatter(list(error_pages_input), [4] * len(error_pages_input), color='r', marker='x', s=100, label="Input XOR Errors")

    # Error Markers for Control Comparison (Orange 'X')
    if error_pages_control:
        plt.scatter(list(error_pages_control), [8] * len(error_pages_control), color='orange', marker='x', s=100, label="Control XOR Errors")

    plt.xlabel("Page Numbers")
    plt.ylabel("Grouping (1=True, 0=False)")
    plt.title("Comparison of Page Grouping with Input & Control")
    plt.yticks([0, 1, 4, 5, 8, 9], ["False (Processed)", "True (Processed)", "False (Input)", "True (Input)", "False (Control)", "True (Control)"])  
    plt.grid(True, linestyle="--", alpha=0.9)
    plt.legend()
    plt.savefig("output/group_comparison_chart.png")
    plt.show()
    print("📊 Graph saved at output/group_comparison_chart.png")

    return error_pages_input, error_pages_control

def calculate_error_percentage(processed_file, input_file, control_file):
    """Calculate the error percentage for both input and control CSVs."""
    
    def extract_pages(file_path):
        """Extract page ranges from the CSV files."""
        pages = set()
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

            for row in rows:
                start_page = int(row[0])
                end_page = int(row[1])
                pages.update(range(start_page, end_page + 1))
        
        return pages

    # Extract Pages
    processed_pages = extract_pages(processed_file)
    input_pages = extract_pages(input_file)
    control_pages = extract_pages(control_file)

    # XOR Errors
    error_pages_input = processed_pages ^ input_pages
    error_pages_control = processed_pages ^ control_pages

    # Error Percentages
    total_pages = len(processed_pages | input_pages | control_pages)
    error_percentage_input = (len(error_pages_input) / total_pages) * 100 if total_pages > 0 else 0
    error_percentage_control = (len(error_pages_control) / total_pages) * 100 if total_pages > 0 else 0

    # Print Analysis
    print("\n=== 📊 Page Matching Analysis (XOR) ===")
    print(f"✅ Error Percentage (Input): {error_percentage_input:.2f}%")
    print(f"✅ Error Percentage (Control): {error_percentage_control:.2f}%\n")

    return error_percentage_input, error_percentage_control

if __name__ == "__main__":
    grouped_sections = group_pages(ocr_data)
    csv_file = save_to_csv(grouped_sections)
    input_csv = "input/sample5_pdf.csv"
    control_csv = "input/control_sample5.csv"

    # Generate Comparison Chart
    error_input, error_control = plot_comparison_chart(csv_file, input_csv, control_csv)

    # Calculate Error Percentage
    error_percentage_input, error_percentage_control = calculate_error_percentage(csv_file, input_csv, control_csv)

    # Output Error Summary
    print("\n🚨 Pages with Errors (Input):", sorted(error_input))
    print("🚨 Pages with Errors (Control):", sorted(error_control))
