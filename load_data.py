# load_data.py
import pandas as pd
import fitz  # PyMuPDF for PDF extraction
import os

# Directory where the data files are stored
DATA_DIR = "data"

# Function to read CSV files (supports large files by reading in chunks)
def read_csv(file_path, chunk_size=10000):
    # Return a generator that yields chunks of data
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

# Function to read Excel files (supports large files by reading in chunks)
def read_excel(file_path, chunk_size=10000):
    # Using ExcelFile to allow processing large files in chunks
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        sheet = excel_file.parse(sheet_name)
        # Read the file in chunks
        for start_row in range(0, len(sheet), chunk_size):
            chunk = sheet.iloc[start_row:start_row + chunk_size]
            yield chunk

# Function to read TXT files (just reads the whole text)
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Extract text from PDF file (reads entire content)
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to handle different file types and return data
def read_file(file_name):
    file_path = os.path.join(DATA_DIR, file_name)  # Create full file path
    
    if file_name.endswith('.csv'):
        return read_csv(file_path)  # Returns a generator for large files
    elif file_name.endswith('.xlsx'):
        return read_excel(file_path)  # Returns a generator for large files
    elif file_name.endswith('.txt'):
        return read_txt(file_path)  # Returns the full text for TXT files
    elif file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_path)  # Returns the full text for PDF files
    else:
        raise ValueError("Unsupported file type")

# Example usage: Loading the hospital_patients.xlsx file
if __name__ == "__main__":
    file_name = "hospital_patients.xlsx"  # Your specific file name
    data_chunks = read_file(file_name)
    
    # For CSV/Excel (chunked data), we can print each chunk
    if isinstance(data_chunks, str):  # For TXT or PDF files
        print(data_chunks[:500])  # Print the first 500 characters of the file
    elif isinstance(data_chunks, pd.DataFrame):  # In case data is already a DataFrame
        print(data_chunks.head())  # Print the first few rows of the DataFrame
    else:
        for chunk in data_chunks:  # For chunked CSV/Excel, iterate through chunks
            print(chunk.head())  # Print the first few rows of each chunk
