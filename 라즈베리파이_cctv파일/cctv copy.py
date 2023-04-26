# -*- coding: utf8 -*-
#! /home/kiki/cctv/cctv-env/bin/python3
######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# 실행 : $ python3 TFLite_detection_webcam.py --modeldir=TFLite_model
# 공부할 것들(TPU, interpreter, )
# /home/kiki/cctv/cctv-env/bin/python3 /home/kiki/cctv/cctv.py & exit 0

import ipaddress
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from uuid import uuid4
import datetime
import RPi.GPIO as GPIO
from pyfcm import FCMNotification
import socket
import shutil

# firebase 정보 
PROJECT_ID = "test-fffec"
CRED = credentials.Certificate("/home/kiki/samba/test-fffec-firebase-adminsdk-jff4b-4b4b114913.json")
DEFAULT_APP = firebase_admin.initialize_app(CRED,{'storageBucket':f"{PROJECT_ID}.appspot.com"})
BUCKET = storage.bucket()
IMAGE_FOLDER = '/home/kiki/cctv/cam_img/'

RUN_LED = 19
IDLE_LED = 26
CCTV_OFF_BTN_PIN = 20
CCTV_ON_BTN_PIN = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(RUN_LED, GPIO.OUT)
GPIO.setup(CCTV_OFF_BTN_PIN, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(IDLE_LED, GPIO.OUT)
GPIO.setup(CCTV_ON_BTN_PIN, GPIO.IN, GPIO.PUD_UP)

# cctv 종료 버튼 콜백함수 등록
GPIO.add_event_detect(CCTV_OFF_BTN_PIN, GPIO.RISING)

# firebase API key
push_service = FCMNotification("AAAAybUThak:APA91bF-h9BdW07E1RM1GCUw7RrGC367viE7N7Z4DekyQV8serT_23Kh3KIL4qa0ojOFYTznUTgX8ifh41r9L_TF_V4CftiDZpa9D_R_lkyOacp5uJbbHF79vgGbSIP_j1lriTFylTJf")
# FCM app token key
registration_id = "c7i2W_imSHGbYHwHQONvLO:APA91bGqW2o-unlbSZllpEl5yGqgg20XA1vSqSCEfKphQP_tFK1517mWiSpCorvZ6tEZ_aaZsx3MAm5xCzJ19TwOjGoa8Bd6ocQIXdJoVx3idbtOgz5Tr8LgOgSCoM62bgnpA-hoOle4"

def idle():
    GPIO.output(IDLE_LED, GPIO.HIGH)
    GPIO.wait_for_edge(CCTV_ON_BTN_PIN, GPIO.RISING, bouncetime=500)
    GPIO.output(IDLE_LED, GPIO.LOW)

def fileUpload(file):
    blob = BUCKET.blob('image_storage/'+file)
    new_tocken = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_tocken}
    blob.metadata = metadata

    file_name = IMAGE_FOLDER+file
    image_type = 'image/png'

    blob.upload_from_filename(filename=file_name, content_type=image_type)

    try:
        sendMessage("CCTV 알림", file, new_tocken)
    except:
        print("send message failed")

def clearAllImage():
    path = IMAGE_FOLDER
    try:
        if os.path.exists(path):
            os.system('rm -rf %s*' % path)
    except:
        print ("image remove err")

# FCM
def sendMessage(title, name, token):
    # https://firebasestorage.googleapis.com/v0/b/
    # test-fffec.appspot.com
    # /o/
    # image_storage%2F
    # cctv_20220523_015349.png
    # ?alt=media
    # &token=e3bc8f5d-240e-404e-a035-15d748f04a74
    # "https://firebasestorage.googleapis.com/v0/b/test-fffec.appspot.com/o/image_storage%2F" + name + "." + image_type + "?alt=media&token=" + file_token
    
    file_name = name
    file_token = str(token)

    new_body = "https://firebasestorage.googleapis.com/v0/b/test-fffec.appspot.com/o/image_storage%2F" + file_name + "?alt=media&token=" + file_token
    
    message ={
        "title" : title,
        "body" : new_body
    }

    push_service.single_device_data_message(registration_id=registration_id, data_message=message)
    #push_service.notify_topic_subscribers(topic_name="test", data_message=message)

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=24):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# 명령 터미널에서 실행 시 입력되는 설정 값들을 파싱해서 저장하는 부분.
# ex) $ python3 TFLite_detection_stream.py '--modeldir' '--streamurl'
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='/model/')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

# 위에서 파싱한 값들을 각 변수별로 나눠 저장한다.
args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
from tflite_runtime.interpreter import Interpreter
if use_TPU:
    from tflite_runtime.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# 현재 경로 + modeldir + 파일이름
# 부팅하고 .sh로 실행할때 뒤에 있는 설정값을 못 불러오는 버그 때문에 고정된 경로로 지정함. 
# 모델 경로(.tflite file)
PATH_TO_CKPT = '/home/kiki/cctv/model/detect.tflite'

# 라벨 경로(.txt)
PATH_TO_LABELS = '/home/kiki/cctv/model/labelmap.txt'

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

def check_internet():
    success_internet = False
    try:
        while success_internet == False:
            ipaddress = socket.gethostbyname(socket.gethostname())
            if ipaddress == "127.0.0.1":
                print("Your not connected to the internet!")
            else:
                print("Your connected to the internet with the IP address " + ipaddress)
                success_internet = True
            time.sleep(1.5)
    except:
        print("internet check err")

# 이미지 캡쳐 후 전송
def excute_capture(vid):
    basename = "cctv"
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    filename = "_".join([basename, suffix])

    # cv2.imwrite(파일이름, 저장할 이미지[,파라미터]) 파일이름에 저장경로 포함, home/pi/image.png
    cv2.imwrite(IMAGE_FOLDER + filename, vid.read())
    fileUpload(filename)

def run_cctv(vid):
    try:
        last_captured_time = 0
        DELAY_CAP_TIME = 2
        
        while True:
            GPIO.output(RUN_LED, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(RUN_LED, GPIO.LOW)

            frame1 = vid.read()

            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            # Acquire frame and resize to expected shape [1xHxWx3]
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            for i in range(len(scores)):
                # 0 = 사람
                if (classes[i] != 0):
                    break

                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    # 사람이고 찍은 지 DELAY_CAP_TIME 초 지났을 때
                    if time.time() - last_captured_time >= DELAY_CAP_TIME:
                        excute_capture(vid)
                        last_captured_time = time.time()

            if GPIO.event_detected(CCTV_OFF_BTN_PIN):
                return
                
    except KeyboardInterrupt:
        clearAllImage()
        vid.stop()
        exit(0)

try:
    while True:
        check_internet()
        idle()
        videostream = VideoStream().start()
        run_cctv(videostream)
        print ("switch off")
        videostream.stop()
        print ("video stream off")
        clearAllImage()
    
finally:
    videostream.stop()
    GPIO.output(IDLE_LED, GPIO.LOW)
    GPIO.output(RUN_LED, GPIO.LOW)
    GPIO.cleanup()
    clearAllImage()
    exit(0)