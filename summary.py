import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from dotenv import load_dotenv
from langdetect import detect
import easyocr
import numpy as np
from PIL import Image
import fitz 
from pdf2image import convert_from_path
import pytesseract
import tempfile
from io import BytesIO

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file or environment variables.")
genai.configure(api_key=API_KEY)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)

def detect_language(text):
    lang = detect(text)
    return lang

def read_image(file, lang):
    image = Image.open(file)
    image_np = np.array(image)  # Convert PIL Image to numpy array
    
    # Language groups
    latin_languages = ['en', 'fr', 'de', 'es', 'it', 'pt']
    cyrillic_languages = ['ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'en']
    ja_ko_zh_languages = ['ja', 'ko', 'zh-cn', 'zh-tw', 'en']
    
    if lang in ['ja', 'ko', 'zh-cn', 'zh-tw']:
        reader = easyocr.Reader(ja_ko_zh_languages)
    elif lang in cyrillic_languages:
        reader = easyocr.Reader(cyrillic_languages)
    else:
        reader = easyocr.Reader(latin_languages)
    
    result = reader.readtext(image_np, detail=0)
    
    text = ' '.join(result)
    return text

def extract_text_from_file(file):
    """Extracts text from various file types."""
    try:
        if file.name.endswith(".pdf"):
            text = ""
            pdf_bytes = file.read()

            if len(pdf_bytes) == 0:
                st.error("Error: Uploaded PDF file is empty.")
                return None

            try:
                # Try PyPDF2 first
                pdf_reader = PdfReader(BytesIO(pdf_bytes))
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except Exception:
                pass  # Proceed to next method if PyPDF2 fails

            if not text:
                try:
                    # Try PyMuPDF if PyPDF2 failed or was empty
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in doc:
                        page_text = page.get_text("text")
                        if page_text.strip():
                            text += page_text + "\n\n"
                except Exception:
                    pass # Proceed to next method if PyMuPDF fails

            if not text:
                try:
                    # Attempt OCR using EasyOCR to extract text from images in PDF
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        img = Image.open(BytesIO(img_bytes))
                        lang = detect_language(read_image(BytesIO(img_bytes), 'en')) #detect language of image.
                        text += read_image(BytesIO(img_bytes), lang) + "\n\n"

                except Exception as e:
                    st.error(f"OCR Failed on PDF: {e}")
                    return None

            return text

        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")  # Handle text encoding
            return text

        elif file.name.endswith(".docx"):
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text

        elif file.name.endswith((".jpg", ".jpeg", ".png")):
            temp_image_text = read_image(file, 'en')  # Use English as a placeholder for detection
            detected_lang = detect_language(temp_image_text)
            text = read_image(file, detected_lang)
            return text

        else:
            st.error("Unsupported file type. Please upload a PDF, TXT, DOCX, JPG, JPEG, or PNG file.")
            return None

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None


def summarize_document(text, prompt_template=None):
    prompt_template = "Summarize the information under each heading:\n\n{text}"
    if not text:
      st.error("No text provided to summarize")
      return None
    try:
        prompt = prompt_template.format(text=text)
        response = model.generate_content(prompt)
        if response.text:
           return response.text
        else:
          st.error(f"No summary generated from Gemini response: {response}")
          return None
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return None


def summarize_pdf(pdf_file):
    extracted_text = extract_text_from_file(pdf_file)
    if not extracted_text:
      return None

    summary = summarize_document(extracted_text)
    if summary:
      return summary
    else:
      return None

def main():
    st.title("DocSummarizer")
    st.write("Upload a document in PDF/Docx/Txt format below and I'll summarize it for you")

    uploaded_file = st.file_uploader("Upload a PDF/Docx/Txt file", type="pdf")

    if uploaded_file:
        with st.spinner("Summarizing the document..."):
            summary = summarize_pdf(uploaded_file)
            if summary:
              st.subheader("Summary:")
              st.write(summary)
            else:
               st.error("Could not create summary")

if __name__ == "__main__":
    main()
