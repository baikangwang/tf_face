from flask import Flask, Response, request, jsonify, render_template
from PIL import Image, ImageDraw
from io import BytesIO
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials
from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import tensorflow as tf
import numpy
import base64
import cStringIO
import os
import json

import resources, classes

import cv2
import sys

tf.app.flags.DEFINE_string('host', 'localhost', 'Prediction host')
tf.app.flags.DEFINE_integer('port', 9000, 'Prediction host port')
tf.app.flags.DEFINE_string('model_name', 'pubfig', 'TensorFlow model name')
tf.app.flags.DEFINE_integer('image_size', 96, 'Edge size of face')
tf.app.flags.DEFINE_integer('image_channels', 3, 'Number of image channels')
tf.app.flags.DEFINE_float('timeout', 5.0, 'Prediction grpc timeout in seconds')
tf.app.flags.DEFINE_integer('max_predictions', 10, 'Maximum number of predictions')

SCOPES = 'https://www.googleapis.com/auth/cloud-platform'
FLAGS = tf.app.flags.FLAGS

flask_app = Flask(__name__, static_url_path='')
vision_svc = None

def get_vision_service():
    print("-initial_cred")
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.path.join(resources.__path__[0], 'vapi-acct.json'), SCOPES)
    print(credentials)
    return discovery.build('vision', 'v1', credentials=credentials)

@flask_app.route('/', methods=['GET'])
def get_index():
    return render_template("index.html",
        results = 'false',
        error = 'false')

@flask_app.route('/', methods=['POST'])
def classify_file():
    # Detect where the face is in the picture
    f = request.files['file']
    image_content = f.read()
    print("+loaded")
    # top, left, bottom, right, rgb = detect_face(image_content)
    top, left, bottom, right, rgb = detect_face_opencv(image_content)
    print("+detected")

    if rgb is not None:
        # Get prediction from model server
        pred = recognise_face(rgb)

        # Find the top classes and scores
        top_pred = pred.argsort()[-FLAGS.max_predictions:][::-1].tolist()
        top_scores = numpy.sort(pred)[-FLAGS.max_predictions:][::-1]
        top_scores = numpy.multiply(top_scores, 100)

        # Draw a rectangle to identify the face
        im = Image.open(BytesIO(image_content))
        dr = ImageDraw.Draw(im)
        dr.rectangle((left, top, right, bottom), outline="green")
        buffer = cStringIO.StringIO()
        im.save(buffer, format="JPEG")

        return render_template('index.html',
            results = 'true',
            error = 'false',
            image_b64 = base64.b64encode(buffer.getvalue()),
            prediction_1 = classes.CLASSES[top_pred[0]],
            prediction_2 = classes.CLASSES[top_pred[1]],
            prediction_3 = classes.CLASSES[top_pred[2]],
            prediction_4 = classes.CLASSES[top_pred[3]],
            prediction_5 = classes.CLASSES[top_pred[4]],
            score_1 = top_scores[0],
            score_2 = top_scores[1],
            score_3 = top_scores[2],
            score_4 = top_scores[3],
            score_5 = top_scores[4])

    return render_template('index.html',
        results = 'false',
        error = 'true')

def detect_face_opencv(image_content):

    (top, left, bottom, right, rgb) = 0, 0, 0, 0, None
    print("+opencv")
    try:

        # initial image object from post form
        # im = Image.open(BytesIO(image_content))


        # Get user supplied values
        # imagePath = sys.argv[1]
        cascPath = "haarcascade_frontalface_default.xml"

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the image
        stream= BytesIO(image_content)
        stream.seek(0)
        img_array = numpy.asarray(bytearray(stream.read()), dtype=numpy.uint8)
        cv2_img_flag=1 # color
        # print(cv2_img_flag)
        image= cv2.imdecode(img_array, cv2_img_flag)
        print("+opencv_readStream")

        cv2.imwrite("test.png",image)
        print("+opencv_imwrite")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )
        print("+opencv_detected")


        if (len(faces) >=1):
            x,y,w,h = faces[0]
            left=x
            top = y
            right = x+w
            bottom = y+h

            # face_im = im.crop((left, top, right, bottom)).resize((FLAGS.image_size, FLAGS.image_size))
            face_im=cv2.resize(image[top:bottom,left:right],(FLAGS.image_size, FLAGS.image_size),cv2.INTER_CUBIC)
            print("+opencv_crop_resized")
            # rgb = numpy.array(face_im.convert('RGB'))
            rgb=  numpy.asarray(face_im)
            rgb = numpy.divide(rgb, 255.0)
            rgb = numpy.expand_dims(rgb, axis=0)

            # numpy.set_printoptions(threshold=numpy.shape(rgb))
            # numpy.array2string(rgb, separator=',', max_line_width=None).replace('\n', '')

            rgb = rgb.astype(numpy.float32)
            print("+rgb_ready")


    except Exception, e:
        print("Something went wrong: %s" % str(e))

    return top, left, bottom, right, rgb

def detect_face(image_content):

    request_dict = [{
        'image': {
            'content': base64.b64encode(image_content)
            #'content': image_content
            },
        'features': [{
            'type': 'FACE_DETECTION',
            'maxResults': 1,
            }]
        }]

    (top, left, bottom, right, rgb) = 0, 0, 0, 0, None

    try:
        vision_svc = get_vision_service()
        print("+initial_vision")
        request = vision_svc.images().annotate(body={
            'requests': request_dict
        })
        print("+build_request")
        response = request.execute()
        print("+executed")

        face_bounds = response['responses'][0]['faceAnnotations'][0]['fdBoundingPoly']['vertices']

        if (len(face_bounds) == 4):
            left = face_bounds[0]['x']
            top = face_bounds[0]['y']
            right = face_bounds[2]['x']
            bottom = face_bounds[2]['y']

            im = Image.open(BytesIO(image_content))
            face_im = im.crop((left, top, right, bottom)).resize((FLAGS.image_size, FLAGS.image_size))
            rgb = numpy.array(face_im.convert('RGB'))
            rgb = numpy.divide(rgb, 255.0)
            rgb = numpy.expand_dims(rgb, axis=0)

            # numpy.set_printoptions(threshold=numpy.shape(rgb))
            # numpy.array2string(rgb, separator=',', max_line_width=None).replace('\n', '')

            rgb = rgb.astype(numpy.float32)


    except Exception, e:
        print("Something went wrong: %s" % str(e))

    return top, left, bottom, right, rgb

def recognise_face(rgb):
    channel = implementations.insecure_channel(FLAGS.host, FLAGS.port)
    print("+channel")
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    print("+stub")

    request = predict_pb2.PredictRequest()
    print("+request")
    request.model_spec.name = FLAGS.model_name
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(rgb,
            shape=[1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels]))
    print("+proto")
    result = stub.Predict(request, FLAGS.timeout)
    print("+predict")
    values = numpy.array(result.outputs['scores'].float_val)

    return values

def main(_):
    flask_app.run(host='127.0.0.1', port=8080)

if __name__ == '__main__':
    # vision_svc = get_vision_service()
    tf.app.run()
