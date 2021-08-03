"""
Flask Web Application
Link 1: https://github.com/seraj94ai/Flask-streaming-Pedestrians-detection-using-python-opencv-
Link 2: https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
Link 3: https://stackoverflow.com/questions/45149420/pass-variable-from-python-flask-to-html-in-render-template/45151521
Link 4: https://stackoverflow.com/questions/11178426/how-can-i-pass-data-from-flask-to-javascript-in-a-template

Debugging things
Script tag position: https://stackoverflow.com/questions/5371047/getelementbyid-returns-null-even-though-the-element-exists?noredirect=1&lq=1
Realtime data update: https://www.youtube.com/watch?v=E0UGGxd2DOo&t=213s
Flask on localhost: https://stackoverflow.com/questions/30554702/cant-connect-to-flask-web-service-connection-refused
"""

import time
import cv2
import requests
import flask
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)
host = ''
test = 0


@app.route('/')
def index():
    global host
    host = flask.request.host_url
    host = host[:-1]
    """Video streaming home page."""
    return render_template('index.html', data=test)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status', methods=['GET', 'POST'])
def get_updates():
    global test
    if request.method == 'GET':
        update = "Calibrating your position " + str(test)
        return jsonify({"status": update})
    elif request.method == 'POST':
        test += 1
        return jsonify({"status": "success!"})


def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while cap.isOpened():
        ret, img = cap.read()
        assert host is not ''
        requests.post(host + '/status', data={})
        if ret:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.1)
        else:
            break
