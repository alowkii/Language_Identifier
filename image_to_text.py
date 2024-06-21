from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
image_path = 'images/image.png'

img = Image.open(image_path)

text = pytesseract.image_to_string(img, lang='ocrb+eng+rus+hin+kor+jap')

with open('input.txt', 'w', encoding='utf-8') as file:
    file.write(text)