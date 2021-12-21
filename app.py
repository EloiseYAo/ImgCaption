from PIL import Image
import base64
import cv2
from io import BytesIO
import os

from flask import Flask, request, redirect, url_for, jsonify
from flask import render_template
from flask_cors import CORS
from flask import g

from models.inference import Inference
from models.test_input import Input

from utils.config import CONFIG
import matplotlib.pyplot as plt

from img_model import Img
import db_config
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config.from_object(db_config)
db = SQLAlchemy()
db.init_app(app)



@app.route('/')
def index():
    rec_imgs = Img.query.all()
    rec_list = []
    for rec_img in rec_imgs:
        rec_list.append(rec_img.to_json())
    rec_len = len(rec_list)
    show_id=rec_len-1
    caption = rec_list[show_id]['ic_caption']

    img_dict = {'img': show_id, 'caption': caption, 'flag': 0}
    return render_template('index.html', img_dict=img_dict, rec_list=rec_list, rec_len=rec_len)


@app.route('/random')
def random():

    # filecount = 0
    # test_img_path=CONFIG.TEST_IMG_PATH
    # for root, dir, files in os.walk(test_img_path):
    #     filecount += len(files)
    # print("test_img下共有%d个文件" % (filecount-1))
    #
    # m = 1
    # inference = Inference(m)
    # imgs, captions = inference.get_caption()
    #
    # img = Image.fromarray(imgs[0], 'RGB')
    # # print(type(img))
    # img_id = filecount
    # img.save('%s/%d.png' % (test_img_path, img_id))
    #
    # img_list = ['none']
    # img_list.append(captions[0])
    # return render_template('index.html', img_list=img_list, len=img_id)

    m = 1
    inference = Inference(m)
    imgs, captions = inference.get_caption()

    image = Image.fromarray(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2BGRA))
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_img = str(base64_str, encoding="utf-8")

    
    img_dict = {'img': base64_img, 'caption': captions[0],'flag':1}

    print(base64_img)
    print(captions[0])

    # plt.imshow(imgs[0])
    # plt.show()
    rec_imgs = Img.query.all()
    rec_list = []
    for rec_img in rec_imgs:
        rec_list.append(rec_img.to_json())
    rec_len = len(rec_list)

    return render_template('index.html', img_dict = img_dict, rec_list=rec_list, rec_len=rec_len)


@app.route('/getImg', methods=['POST'])
def getInput():
    img = request.files.getlist('img')[0]
    print(img)
    num=Img.query.count()
    path = '%s/%d.png' % (CONFIG.TEST_IMG_PATH, num)
    img.save(path)

    inference = Input(path=path)
    img, caption = inference.get_caption()

    img_insert = Img(ic_id=num, ic_caption=caption)
    db.session.add(img_insert)
    db.session.commit()

    rec_imgs = Img.query.all()
    rec_list = []
    for rec_img in rec_imgs:
        rec_list.append(rec_img.to_json())
    rec_len = len(rec_list)

    # img_dict = {'img': num, 'caption': caption, 'flag':0}

    # return redirect(url_for('showImg', caption=caption))
    img_dict = {'img': num, 'caption': caption, 'flag': 0}
    return render_template('index.html', img_dict=img_dict, rec_list=rec_list, rec_len=rec_len), 401


# @app.route('/showImg/?<caption>')
# def showImg(caption):
#     num=2
#     img_dict = {'img': num, 'caption': caption, 'flag':0}
#     return render_template('index.html', img_dict = img_dict), 401


@app.route('/list')
def list():
    imgs = Img.query.all()
    print(imgs)
    imgs_output = []
    for img in imgs:
        imgs_output.append(img.to_json())
    return jsonify(imgs_output)

@app.route('/list/<int:id>')
def find_caption(id):
    rec_imgs = Img.query.all()
    rec_list = []
    for rec_img in rec_imgs:
        rec_list.append(rec_img.to_json())
    rec_len = len(rec_list)

    caption=rec_list[id]['ic_caption']
    img_dict = {'img': id, 'caption': caption, 'flag': 0}
    return render_template('index.html', img_dict=img_dict, rec_list=rec_list, rec_len=rec_len)


if __name__ == '__main__':
    app.run()
    app.debug = True



