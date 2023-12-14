from matplotlib import text
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import os
from os import listdir
from flask import *
import requests

app = Flask(__name__)

def get_country_name(country_code):
    f = open('./codes.json')
    country_data = json.load(f)
    f.close()

    if country_code not in country_data:
        return "Error: Unable to find country"
    return country_data.get(country_code)

# Following deep learning approach for comparing two images was copied from the following link: 
# https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_country/', methods=['POST'])
def find_country():
    user_image = request.files['user_image']
    user_image.save('./test_images/user_upload.png')
    user_image = './test_images/user_upload.png'
    error_message = "ERROR: CANNOT FIND A MATCH, PLEASE UPLOAD A CLEARER PHOTO"
    if not user_image:
        return render_template('index.html', country_name=error_message, flag=None)
    
    MINIMUM_THRESHOLD = 70.00
    folder_dir = "./flag_icons"
    for images in os.listdir(folder_dir):
        curr_threshold = 85.00
        similarity = generateScore(f'{folder_dir}/{images}', user_image)
        while curr_threshold >= MINIMUM_THRESHOLD:
            if similarity >= curr_threshold:
                country_code = images.split('.')[0]
                country_name = get_country_name(country_code)
                flag = f'https://flagcdn.com/h240/{country_code}.png'
                return render_template('index.html', country_name=country_name, flag=flag)
            curr_threshold -= 5.00
    return render_template('index.html', country_name=error_message, flag=None)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)