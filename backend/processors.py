# processors.py
import re

def clean_text(text_input):
    def transform(text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
        # text = re.sub(r'\d+', '[NUM]', text)
        text = re.sub(r'\b\d{16,}\b', '[LONGNUM]', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return [transform(item) for item in text_input]
    # return text_series.apply(transform)