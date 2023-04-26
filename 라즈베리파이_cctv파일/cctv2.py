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

# THX Evan!
# 사람을 발견했을 때 사진을 찍어 firebase로 전송
# 실행 : $ python3 TFLite_detection_webcam.py --modeldir=TFLite_model
# hdmi가 연결되어 있지 않으면 imshow에서 오류 생기므로 해당 부분 수정할 것.
# 공부할 것들(TPU, interpreter, )
# /home/kiki/cctv/cctv-env/bin/python3 /home/kiki/cctv/cctv.py & exit 0

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
import subprocess

# firebase 정보 
PROJECT_ID = "test-fffec"
cred = credentials.Certificate("/home/kiki/samba/test-fffec-firebase-adminsdk-jff4b-4b4b114913.json")
default_app = firebase_admin.initialize_app(cred,{'storageBucket':f"{PROJECT_ID}.appspot.com"})
bucket = storage.bucket()
IMAGE_FOLDER = '/home/kiki/cctv/cam_img/'

last_captured_time = 0
DELAY_CAP_TIME = 3
STATE_LED = 19
CCTV_OFF_BTN_PIN = 20
start_btn_pin = 16
led_pin = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(STATE_LED, GPIO.OUT)
GPIO.setup(CCTV_OFF_BTN_PIN, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(start_btn_pin, GPIO.IN, GPIO.PUD_UP)

# 종료 절차 실행
def cctvOff(self):
    try:
        clearAllImage()
    finally:
        for n in range(0,3):
            GPIO.output(STATE_LED, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(STATE_LED, GPIO.LOW)
            time.sleep(0.5)

        print("cctv off...")
        cv2.destroyAllWindows()


# 이미지 파일 업로드
def fileUpload(file):
    blob = bucket.blob('image_storage/'+file)
    new_tocken = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_tocken}
    blob.metadata = metadata

    # 업로드 파일
    blob.upload_from_filename(filename=IMAGE_FOLDER+file, content_type='image/png')

# 이미지 파일 제거 
def clearAllImage():
    path = IMAGE_FOLDER
    os.system('rm -rf %s/*' % path)

# def btn_callback():
    

# 이미지 캡쳐 후 전송
def excute_capture():
    basename = "cctv"
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'
    filename = "_".join([basename, suffix])

    # cv2.imwrite(파일이름, 저장할 이미지[,파라미터]) 파일이름에 저장경로 포함, home/pi/image.png
    cv2.imwrite(IMAGE_FOLDER + filename, videostream.read())
    fileUpload(filename)

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
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

# GPIO.add_event_detect(start_btn_pin, GPIO.RISING, callback=btn_callback, bouncetime=500)
# GPIO.add_event_detect(CCTV_OFF_BTN_PIN, GPIO.RISING, callback=cctvOff, bouncetime=300)

def standby():
    print('cctv stand by')
    GPIO.output(led_pin, GPIO.HIGH)
    try:
        while True:
            if (GPIO.input(start_btn_pin) == False):
                GPIO.output(led_pin, GPIO.LOW)
                s()
            
            time.sleep(0.5)
    finally:
        cctvOff()

def s():

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

# TensorFlow 라이브러리를 넣는 부분. (나중에 사용하는 코드만)
# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       


# 현재 작업 경로를 불러오고 파싱한 값에 맞춰 모델 파일과 라벨 파일을 불러오기.
# Get path to current working directory
# CWD_PATH = os.getcwd()

# 현재 경로 + modeldir + 파일이름
# 부팅하고 .sh로 실행할때 뒤에 있는 설정값을 못 불러오는 버그 때문에 고정된 경로로 지정함. 
# 모델 경로(.tflite file)
# PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_CKPT = '/home/kiki/cctv/model/detect.tflite'

# 라벨 경로(.txt)
# PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
PATH_TO_LABELS = '/home/kiki/cctv/model/labelmap.txt'

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

    try:
        while True:
            GPIO.output(STATE_LED, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(STATE_LED, GPIO.LOW)
            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            frame1 = videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # 라벨링 및 ROI 표시하는 부분. confidence가 50 ~ 100%일때 발견된 것으로 처리.
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    # ROI 표시
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label 화면 필요 없으면 라벨링도 삭제
                    object_name = int(classes[i]) # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    # 0 = 사람
                    # 사람이고 찍은 지 DELAY_CAP_TIME 초 지났을 때
                    if ((0 == int(classes[i])) and (time.time() - last_captured_time >= DELAY_CAP_TIME)):
                        excute_capture()
                        last_captured_time = time.time()

            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
            cv2.imshow('Object detector', frame)

            # fps 계산
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            if (cv2.waitKey(1) == ord('q')):
                break
    except:
        cctvOff()

while True:    
    standby()