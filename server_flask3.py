from flask import Flask, request, jsonify

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
    image = request.files['file']
    image_string = base64.b64encode(image.read())
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    shape = "동그라미"
    color = "검은색"
    return jsonify(
        shape=shape,
        color=color
    )


# 앱에서 사용자가 데이터를 수정해주면 그 정보를 바탕으로 크롤링 하는 부분
@app.route('/crawl', methods=['POST', 'GET'])
def crawl():
    shape = request.form['shape']
    color = request.form['color']
    text = request.form['text']
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    medicine_image = ['https://terms.naver.com/entry.naver?docId=2141123&cid=51000&categoryId=51000', 'https://terms.naver.com/entry.naver?docId=2140285&cid=51000&categoryId=51000',
                      'https://terms.naver.com/entry.naver?docId=2141235&cid=51000&categoryId=51000', 'https://terms.naver.com/entry.naver?docId=2135883&cid=51000&categoryId=51000',
                      'https://terms.naver.com/entry.naver?docId=2133029&cid=51000&categoryId=51000', 'https://terms.naver.com/entry.naver?docId=2141393&cid=51000&categoryId=51000',
                      'https://terms.naver.com/entry.naver?docId=2160528&cid=51000&categoryId=51000', 'https://terms.naver.com/entry.naver?docId=2134728&cid=51000&categoryId=51000',
                      'https://terms.naver.com/entry.naver?docId=2140277&cid=51000&categoryId=51000', 'https://terms.naver.com/entry.naver?docId=2157828&cid=51000&categoryId=51000']
    link = [
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803233503184_NG6OQL73C.jpg/A11ABBBBB090302.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_9/20171126022112845_RQZOH3G3T.jpg/A11A4290B001503.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210711235422236_PD51TQPKB.jpg/A11ABBBBB116801.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803225930472_0XGWLLLLZ.jpg/A11A0950A004301.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_16/20180819013125082_JJAW30WQP.jpg/A11A0100A008003.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803233920686_HDSPUX76E.jpg/A11ABBBBB150302.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210804021504316_ZHU7552MX.jpg/A11AOOOOO977601.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803224746772_QYJFJKB0I.jpg/A11A0500A011203.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803232831801_LZKIU0UUM.jpg/A11A4270A001401.jpg?type=m250&wm=N',
        'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210804014630115_F9O8JS53Q.jpg/A11AOOOOO547501.jpg?type=m250&wm=N']
    medicine_name = ['인데놀정10mg', '소론도정', '마그밀정', '보나링에이정', '후라시닐정', '리피토정10mg', '하루날디정0.2mg', '슈다페드정', '무코스타정','렉사프로정5mg']
    # 서버에서 string값 두개 리턴하는 코드 여기다가 넣어야함.
    return jsonify(
        medicine_image=medicine_image,
        link=link,
        medicine_name=medicine_name
    )


if __name__ == '__main__':
    app.run(debug=True,host="112.186.193.163",port=5000)
