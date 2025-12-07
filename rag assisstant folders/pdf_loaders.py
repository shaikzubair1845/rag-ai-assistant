import fitz

class PDFLoader:
    def load(self, pdf_path):
        """
        Loads a PDF file and returns extracted text.
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
