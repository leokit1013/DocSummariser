import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

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


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
      text = ""
      pdf_reader = PdfReader(pdf_file)
      for page in pdf_reader.pages:
          text += page.extract_text()
      return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
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
    extracted_text = extract_text_from_pdf(pdf_file)
    if not extracted_text:
      return None

    summary = summarize_document(extracted_text)
    if summary:
      return summary
    else:
      return None

def main():
    st.title("DocSummarizer")
    st.write("Upload a document in PDF format below and I'll summarize it for you")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

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