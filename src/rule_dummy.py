import os
import json
import logging
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sample="3"
input_dir="output/"
csv_file = f"input/control_sample{sample}.csv"


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
    
    return 1

def date_signal(prevPage,currPage):
    return 1

def heading_signal(prevPage,currPage):
    return 1

#def page_info_general(prevPage,currPage):
    #return 1

def isContinous(prevPage,currPage):
    continuity_weight = 0
    continuity_threshold = 0.5
    page_info_weight = 0.4
    date_weight = 0.3
    heading_weight = 0.3
    #page_info_general_weight = 0.0
    continuity_weight += page_info_signal(prevPage,currPage)*page_info_weight
    continuity_weight += date_signal(prevPage,currPage)*date_weight
    continuity_weight += heading_signal(prevPage,currPage)*heading_weight
    #continuity_weight += page_info_general(prevPage,currPage)*page_info_general_weight
    
    
    return 0 if 1- continuity_weight < continuity_threshold else 1



def processSignals(processedvector,data,num_pages):
    processedvector[0] = 1
    for i in range (1, num_pages):
        processedvector[i] = isContinous(data["pages"][i-1],data["pages"][i])
    return processedvector

def main():
    data,num_pages =  readInputJson()
    controlvector = [0]*num_pages
    controlvector = read_input_csv(controlvector)
    print(f"Control Vector : {controlvector}")
    processedvector = [1]*num_pages
    processedvector = processSignals(processedvector,data,num_pages)
    print(f"Processed Vector : {processedvector}")
    errorCounter=0
    errorlist=[]
    errorString=[]
    for i in range(0, num_pages):
        print(f"Control Vector : Page {i+1}: {controlvector[i]}\t\tProcessed Vector : Page {i+1}: {processedvector[i]}\t\t Error = {controlvector[i]^processedvector[i]}")
        if controlvector[i]^processedvector[i] == 1:
            errorCounter+=1
            errorString.append("DC") if controlvector[i] else  errorString.append("CD")
            print(f"Error on Page {i+1}")
            errorlist.append(i+1)
    print(f"Error Pages : {errorlist}")
    print(f"Total Error Percentage : {errorCounter/num_pages*100:.2f}%")

if __name__ == "__main__":
    main()
