import server
import skimage.draw
"""
model = "C:/Users/user/PycharmProjects/maskrcnn-custom/logs/mask_rcnn_experiment_0096.h5"
# img_link = 'C:/Users/user/PycharmProjects/maskrcnn-custom/KakaoTalk_20210927_204431533_15.jpg'
img_link = "C:/Users/user/PycharmProjects/maskrcnn-custom/KakaoTalk_20210927_204431533_19.jpg"
# img_link = "C:/Users/user/PycharmProjects/maskrcnn-custom/다운로드(4)(1).jpg"
# img_link = "C:/Users/user/PycharmProjects/maskrcnn-custom/다운로드(4)(1).jpg"

image = skimage.io.imread(img_link)
print(image)
print(type(image))

color_result, shape_result = server.get_img_inform(model, image)
# color_result = 'light_purple'
# shape_result = 'ellipse'
text_result = ''
print(color_result, shape_result)
# total = server.crawling_get_link_img_name(shape_result, color_result, text_result=text_result)

# print(color_result, shape_result)
# print(crawling_link)
# print(crawling_img)
# print(crawling_name)

# for i in total:
#     print(i)

"""

from flask import Flask, request, jsonify
import server
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import skimage.draw
import imageio
import cv2
from io import BytesIO
app = Flask(__name__)


@app.route('/')
def home():
    return '연결 성공!'


# localhost:5000/image
# 갤러리에서 이미지를 보내면 분석을 해서 다시 리턴해주는 부분.
# 분석하는 코드는 없음.
@app.route('/image', methods=['POST', 'GET'])
def image():
    import base64
    import json
    # image = request.files['file']
    img = request.args.get('img')
    print(img)
    print(type(img))
    imgdata = base64.b64decode(str(img))
    print(imgdata)
    print(type(imgdata))

    # filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    # with open("imageToSave.jpg", "wb") as fh:
    #     fh.write(base64.decodebytes(imgdata))
    filename = "yourfile.jpg"
    with open(filename, "rb") as fid:
        data = fid.read()
    b64_bytes = base64.b64encode(data)

    # im = Image.open(BytesIO(imgdata))
    # im.show()



    # image_string = base64.b64encode(image.read())
    # tmp = Image.open(image)
    # tmp.show()
    # tmp.save('test.jpg')
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    # if image:
    #     filename = secure_filename(image.filename)
    model = "C:/Users/user/PycharmProjects/maskrcnn-custom/logs/mask_rcnn_experiment_0096.h5"
    #1. 이미지 저장후 보내기
    # color, shape = server.get_img_inform(model, "C:/Users/user/PycharmProjects/maskrcnn-custom/23.python_server/test.jpg")
    #2. 저장없이 보내기
    # color, shape = server.get_img_inform(model, image)
    # print(shape, color)
    shape = "circle"
    color = "black"
    return jsonify(
        shape=shape,
        color=color
    )

# localhost:5000/image
# 갤러리에서 이미지를 보내면 분석을 해서 다시 리턴해주는 부분.
# 분석하는 코드는 없음.
@app.route('/image1', methods=['POST', 'GET'])
def image2():
    import base64
    import json
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    shape = "circle"
    color = "black"
    return jsonify(
        shape=shape,
        color=color
    )


# 앱에서 사용자가 데이터를 수정해주면 그 정보를 바탕으로 크롤링 하는 부분
@app.route('/crawl', methods=['POST', 'GET'])
def crawl():
    shape = request.args.get('shape')
    color = request.args.get('color')
    text = request.args.get('text')
    # image_string 을 분석기에 보내서 분석하는 코드가 필요

    # shape = 'circle'
    # color = 'white'
    print(shape, color, text)
    total = server.crawling_get_link_img_name(shape, color, text)

    # tmp = {'medicine_name' : '인데놀정10mg',
    #        'medicine_image' : 'https://terms.naver.com/entry.naver?docId=2141123&cid=51000&categoryId=51000',
    #        'link' : 'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803233503184_NG6OQL73C.jpg/A11ABBBBB090302.jpg?type=m250&wm=N'}
    # tmp2 = {'medicine_name' : '소론도정',
    #         'medicine_image' : 'https://terms.naver.com/entry.naver?docId=2140285&cid=51000&categoryId=51000',
    #         'link' : 'https://dbscthumb-phinf.pstatic.net/3323_000_9/20171126022112845_RQZOH3G3T.jpg/A11A4290B001503.jpg?type=m250&wm=N'}
    # total = [tmp,tmp2]

    # 서버에서 string값 두개 리턴하는 코드 여기다가 넣어야함.
    return jsonify(
        total = total
    )


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True,host="220.125.156.59",port=5000)
