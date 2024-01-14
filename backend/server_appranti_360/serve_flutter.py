from flask import Flask
from flask import send_from_directory
from flask import render_template

app = Flask(__name__, template_folder='web')


FLUTTER_WEB_APP = 'web'


@app.route('/')
def render_page():
    return render_template('/index.html')


@app.route('/web/')
def render_page_web():
    return render_template('index.html')


@app.route('/web/<path:name>')
def return_flutter_doc(name):
    datalist = str(name).split('/')
    DIR_NAME = FLUTTER_WEB_APP

    if len(datalist) > 1:
        for i in range(0, len(datalist) - 1):
            DIR_NAME += '/' + datalist[i]

    return send_from_directory(DIR_NAME, datalist[-1])


if __name__ == '__main__':
    # https://github.com/Algure/flask_flutter_server
    # https://betterprogramming.pub/serving-flutter-web-applications-with-python-flask-c60ab5fc3fc1
    app.run(port=5050)
