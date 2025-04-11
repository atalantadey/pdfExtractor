import os
import json
import logging
import csv
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sample="3"
input_dir="output/"
csv_file = f"input/control_sample{sample}.csv"


def longest_common_substring_length(s1, s2):
    """Return the longest common substring between two strings (case-insensitive)."""
    if not s1 or not s2:
        return ""
    s1, s2 = s1.lower(), s2.lower()
    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest = ""
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > len(longest):
                    longest = s1[i - m[i][j]:i]
    return longest



def readInputJson():
    with open(input_dir + f"ocr_results{sample}.json") as file:
        data = json.load(file)
    num_pages = len(data["pages"])
    return data,num_pages

def read_input_csv(controlvector):
    with open(csv_file , "r") as f:
        reader = csv.reader(f)
        #next(reader, None)  # Skip header if present
        for row in reader:
            if len(row) >= 2:
                try:
                    start_page = int(row[0])
                    if controlvector:
                        controlvector[start_page-1] = 1
                except ValueError as e:
                    logging.warning(f"Invalid row in CSV: {row} - {e}")
    return controlvector
def page_info_signal(prevPage,currPage):
    page_info_strict_curr = currPage["signals"].get("page_info_strict")
    page_info_strict_prev = prevPage["signals"].get("page_info_strict")
    if page_info_strict_prev and page_info_strict_curr:
        prev_pn, prev_total = page_info_strict_prev
        curr_pn, curr_total = page_info_strict_curr
        print(f"Prev Page Number: {prev_pn}, Prev Total Pages: {prev_total}")
        print(f"Curr Page Number: {curr_pn}, Curr Total Pages: {curr_total}")
        # Continuous if same total and current page is next
        if prev_total == curr_total and prev_pn + 1 == curr_pn:
            return 1  # Continuous
        # New subdocument if current page is 1
        if curr_pn == 1 and 1 < curr_total <= 20:
            return 0  # New subdocument
    return 0  # Discontinuous
    

def date_signal(prevPage,currPage):
    """Check if dates are within 32 days."""
    prev_dates = prevPage["signals"].get("date")
    curr_dates = currPage["signals"].get("date")
    if prev_dates and curr_dates:
        for d1 in prev_dates:
            if d1 in curr_dates:
                print(f"Matching date found: {d1}")
                return 1  # Dates are close
    return 0  # No close dates

def heading_signal(prevPage,currPage):
    heading = currPage["signals"].get("heading")
    heading_prev = prevPage["signals"].get("heading")
    if heading_prev and heading:
        lcs = longest_common_substring_length(heading_prev, heading)
        if len(lcs) > 0.8 * min(len(heading_prev), len(heading)):
            print(f"Heading {heading_prev} and {heading} ::::: are similar")
            return 1  # Headings are similar
    return 0  # Headings differ

#def page_info_general(prevPage,currPage):
    #return 1

def isContinous(prevPage,currPage):
    continuity_weight = 0
    continuity_threshold = 0.3
    page_info_weight = 0.4
    date_weight = 0.3
    heading_weight = 0.3
    continuity_weight += page_info_signal(prevPage,currPage)*page_info_weight
    continuity_weight += date_signal(prevPage,currPage)*date_weight
    continuity_weight += heading_signal(prevPage,currPage)*heading_weight
    if page_info_signal(prevPage,currPage) or date_signal(prevPage,currPage) or heading_signal(prevPage,currPage):
        print(f"Page Info Weight : {page_info_signal(prevPage,currPage)*page_info_weight}\tDate Weight : {date_signal(prevPage,currPage)*date_weight}\tHeading Weight : {heading_signal(prevPage,currPage)*heading_weight}\t\tContinuity Weight : {continuity_weight}\n\n")
    return 0 if continuity_weight >= continuity_threshold else 1

def processSignals(processedvector,data,num_pages):
    processedvector[0] = 1# first line is always discontinous
    for i in range (1, num_pages):
        page_info = data["pages"][i]
        page_info_prev = data["pages"][i-1]
        page_number = page_info["page_number"]
        print(f"Page Number : {page_number}")
        processedvector[i] = isContinous(page_info_prev,page_info)
        
    return processedvector

def main():
    data,num_pages =  readInputJson()#reading the JSON file
    #Control Vector 
    controlvector = [0]*num_pages
    controlvector = read_input_csv(controlvector)
    print(f"Control Vector : {controlvector}")
    #Processed Vector
    processedvector = [1]*num_pages
    processedvector = processSignals(processedvector,data,num_pages)
    print(f"Processed Vector : {processedvector}")
    
    #Error Calculation
    errorCounter=0
    errorlist=[]
    errorString=[]
    for i in range(0, num_pages):
        #print(f"Control Vector : Page {i+1}: {controlvector[i]}\t\tProcessed Vector : Page {i+1}: {processedvector[i]}\t\t Error = {controlvector[i]^processedvector[i]}")
        if controlvector[i]^processedvector[i] == 1:
            errorCounter+=1
            errorString.append("DC") if controlvector[i] else  errorString.append("CD")
            #print(f"Error on Page {i+1}")
            errorlist.append(i+1)
    print(f"Error Pages : {errorlist}")
    print(f"Total Error Percentage : {errorCounter/num_pages*100:.2f}%")

if __name__ == "__main__":
    main()
