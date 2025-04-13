import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "Self-Steering Paper.pdf"
    try:
        text = extract_text_from_pdf(pdf_path)
        print(text)
    except Exception as e:
        print(f"Error: {e}")
