# PDF Subdocument Extraction and Error Analysis

Current Tasks : 
*To account for any redudant code in the RuleEngine_v5.py and minimize the size of the code manually 
*To update and play around with the variables and threshold values to figure out and minimize the error percentage


This repository contains Python code for automatically extracting potential subdocuments from a PDF file and analyzing the accuracy of this extraction by comparing it against manually created ground truth (input and control CSV files).

## Overview

The script `ruleEngine.py` performs the following steps:

1.  **OCR Processing:** If `output/ocr_results.json` doesn't exist, it performs Optical Character Recognition (OCR) on the input PDF (`input/sample5.pdf`) using PyMuPDF and a basic image-to-text conversion (note: the actual OCR implementation might need a more robust library like Tesseract for better accuracy). The OCR results (text content and layout information per page) are saved to `output/ocr_results.json`.
2.  **Page Grouping:** It analyzes the OCR results to automatically group pages into potential subdocuments based on a set of rules and priorities:
    * **Priority 1:** Pages identified with a "Page 1 of X" pattern (where X >= 2).
    * **Priority 2:** Consecutive pages that share at least one common date.
    * **Priority 3:** Adjacent pages with highly similar headings (based on the length of the longest common substring).
    * **Priority 4:** Pages with a high continuity weight (indicating textual flow or formatting consistency).
    * **Priority 5:** Adjacent pages with generally similar headings (using SequenceMatcher ratio).
    * **Lowest Priority:** Individual pages are treated as separate subgroups if none of the above criteria are met.
3.  **Saving Processed Groups:** The identified page groups (start page, end page, heading, start date, end date) are saved to `output/processed_subdocument_sample3.csv`.
4.  **Error Analysis:** It compares the automatically generated page groups in `output/processed_subdocument_sample3.csv` with the ground truth provided in `input/sample5.csv` (referred to as "Input") and `input/control_sample5.csv` (referred to as "Control"). It calculates the percentage of pages that are grouped differently by the script compared to each of the ground truth files.
5.  **Visualization:** A comparison chart (`output/group_comparison_chart.png`) is generated to visually represent the page groupings from the script, the input CSV, and the control CSV. Error pages (where the script's grouping differs from the ground truth) are also highlighted.
6.  **Error Page PDF:** A PDF file (`output/error_pages.pdf`) is created, containing the pages where the script's grouping differs from the `control_sample5.csv`. For each error page, the previous, current, and next pages are combined into an image within the PDF to provide context.

## Further Improvements
# The current code provides a basic framework for subdocument extraction and error analysis. Potential areas for improvement include:

*Robust OCR: Integrating a more accurate OCR library like Tesseract for better text extraction.
*Advanced Heading Analysis: Implementing techniques like stop word removal, stemming, and synonym recognition for more accurate heading comparisons.
*Content-Based Analysis: Incorporating content similarity or topic modeling to improve grouping, especially in the absence of clear headings or strong continuity signals.
*Visual Layout Analysis: Analyzing the visual structure of pages (e.g., consistent formatting, section numbering) as additional grouping criteria.
*Machine Learning: Training a model to predict page groupings based on various features extracted from the PDF.
*More Comprehensive Evaluation: Implementing more sophisticated evaluation metrics (e.g., precision, recall, F1-score at the group level).
