from flask import Flask, render_template, request
import re, logging

import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['pt'])

@app.route('/')
def homepage():
    return 'API ON'

@app.route('/recognize-license-plate/')
def recognize_license_plate():
    args = request.args
    url_image = args.get('url_image')
    result = reader.readtext(url_image)
    txt = 'Não Encontrado'
    for itemValue in result:
        if re.search("[A-Z]{3}\d{1}[A-Z]\d{2}|[A-Z]{3}\d{4}", itemValue[1].replace(' ', '')) != None:
            txt = itemValue[1].replace(' ', '')
    return page(txt, url_image)
        
def page(valueLicensePlate: str, url: str):
    return render_template("index.html", user_image = url, valuePlate = valueLicensePlate)

app.run(host='0.0.0.0')