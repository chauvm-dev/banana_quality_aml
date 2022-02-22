import os
import cv2
from flask import Flask, render_template, Response, request, current_app, g
from flask.json import JSONEncoder
from flask_cors import CORS
import numpy as np
##from flask_bcrypt import Bcrypt
##from flask_jwt_extended import JWTManager

from bson import json_util, ObjectId
from datetime import datetime, timedelta
import torch
# from mflix.api.movies import movies_api_v1
from mflix.api.users import users_api_v1
import tensorflow as tf
import random
from flask_pymongo import PyMongo
from werkzeug.local import LocalProxy

detect_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
banana_model = tf.keras.models.load_model("banana_1702_model.h5")
class MongoJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, ObjectId):
            return str(obj)
        return json_util.default(obj, json_util.CANONICAL_JSON_OPTIONS)

def get_db():
    """
    Configuration method to return db instance
    """
    db = getattr(g, "_database", None)

    if db is None:

        db = g._database = PyMongo(current_app).db
       
    return db


def upload_image(path_raw, path_done):
    img_doc = {'pathRaw':path_raw, 'pathDone':path_done}
    db.images.insert_one(img_doc)
    return img_doc


# Use LocalProxy to read the global db instance with just `db`
db = LocalProxy(get_db)


def create_app():

    APP_DIR = os.path.abspath(os.path.dirname(__file__))
    STATIC_FOLDER = os.path.join(APP_DIR, 'build/static')
    TEMPLATE_FOLDER = os.path.join(APP_DIR, 'build')

    app = Flask(__name__, static_folder=STATIC_FOLDER,
                template_folder=TEMPLATE_FOLDER,
                )
    CORS(app)
    app.json_encoder = MongoJsonEncoder
    # app.register_blueprint(movies_api_v1)
    # app.register_blueprint(users_api_v1)

    # @app.route('/', defaults={'path': ''})
    # @app.route('/<path:path>')
    # def serve(path):
    #     return render_template('index.html')


    

    def gen_frames():  
        camera = cv2.VideoCapture(0)
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                results = detect_model(frame)
                labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

                n = len(labels)

                x_shape, y_shape = frame.shape[1], frame.shape[0]

                for i in range(n):
                    row = cord[i]

                    if row[4] >= 0.2:
                        if((int(labels[i])) == int(46)):
                            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] *
                                                                                            x_shape), int(row[3] * y_shape)
                            bgr = (0, 255, 0)

                            crop_frame = frame[y1:y2, x1:x2]
                        

                            crop_frame = cv2.resize(crop_frame, (128, 128))
                            crop_frame = crop_frame / 255.0

                            image_tensor = tf.convert_to_tensor(crop_frame, dtype=tf.float32)
                            image_tensor = tf.expand_dims(image_tensor, 0)

                            prediction = banana_model.predict(image_tensor)

                            prediction_final = np.argmax(prediction)

                            prediction_final = 'Bad' if prediction_final == 0 else 'Good'

                            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

                            cv2.putText(frame,
                                        'Quality: ' + f'{prediction_final}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/image')
    def image():
        return render_template('image.html')

    @app.route("/submit", methods=['GET', 'POST'])
    def get_output():
        if request.method == 'POST':
            img = request.files['my_image']
            random_path = random.randrange(00000000, 99999999, 8)
            img_path_raw = STATIC_FOLDER + '/raw/' +  str(random_path) + '_' + img.filename 
            img_path_done = STATIC_FOLDER + '/done/' + str(random_path) + '_' +  img.filename
            img_path_response = 'static/done/' + str(random_path) + '_' + img.filename

            img.save(img_path_raw)

            frame = cv2.imread(img_path_raw)

            results = detect_model(frame)
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

            n = len(labels)

            x_shape, y_shape = frame.shape[1], frame.shape[0]

            for i in range(n):
                row = cord[i]

                if row[4] >= 0.2:
                    if((int(labels[i])) == int(46)):
                        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] *
                                                                                        x_shape), int(row[3] * y_shape)
                        bgr = (0, 255, 0)

                        crop_frame = frame[y1:y2, x1:x2]
                    

                        crop_frame = cv2.resize(crop_frame, (128, 128))
                        crop_frame = crop_frame / 255.0

                        image_tensor = tf.convert_to_tensor(crop_frame, dtype=tf.float32)
                        image_tensor = tf.expand_dims(image_tensor, 0)

                        prediction = banana_model.predict(image_tensor)

                        prediction_final = np.argmax(prediction)

                        prediction_final = 'Bad' if prediction_final == 0 else 'Good'

                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                        print(x1,y1,x_shape,y_shape)
                        cv2.putText(frame,
                                    'Quality: ' + f'{prediction_final}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

            
            cv2.imwrite(img_path_done, frame)
            upload_image(img_path_raw, img_path_done)
        return render_template("image.html", img_path=img_path_response)
                

    return app
