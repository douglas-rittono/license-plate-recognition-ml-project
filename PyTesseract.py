import pytesseract
import shutil
import os
import random
import requests
try:
 from PIL import Image
except ImportError:
 import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract\\tesseract.exe"
)

listImage = ['Placa-FQF9941', 
             'Placa-HQW5678', 
             'Placa-Mercosul-ABC1B34', 
             'Placa-Mercosul-BRA0S17', 
             'Placa-Mercosul-BRA2O20', 
             'Placa-Mercosul-POX4G21',
             'Placa-Mercosul-POX4G21-So-Placa', 
             'Placa-Mercosul-QRM7E33', 
             'Placa-Mercosul-RHA0A01',
             'Placa-QOZ1774']
imageFormat = 'jpg'
imageRootUrl = 'https://raw.githubusercontent.com/douglas-rittono/license-plate-recognition-ml-project/main/Images/'

for item in listImage:
  image_url = f'{ imageRootUrl }{ item }.{ imageFormat }'
  extractedInformation = pytesseract.image_to_string(Image.open(requests.get(image_url, stream=True).raw))
  print(f'Image: { item } Valor: {extractedInformation.strip()}')
  extractedBox = pytesseract.image_to_boxes(Image.open(requests.get(image_url, stream=True).raw))
  print(f'Image: { item } Box: { extractedBox }')