# -*- coding: utf8 -*-
#! /home/kiki/cctv/cctv-env/bin/python3
import time
import RPi.GPIO as GPIO
import subprocess
import sys
import os

start_btn_pin = 16
led_pin = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(start_btn_pin, GPIO.IN, GPIO.PUD_UP)

def btn_callback(self):
    start_cctv()

def start_cctv():
    print('cctv start')

    for n in range(0,5):
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(0.2)

    GPIO.output(led_pin, GPIO.LOW)
    process = subprocess.Popen(["/home/kiki/cctv/cctv-env/bin/python3", "/home/kiki/cctv/cctv.py"], shell=False)
    process.wait()
    pid = process.pid
    os.system('sudo kill -9 %s' %pid)

try:
    print("standby.py executed.")
    time.sleep(1)
    print("-----press CCTV_ON button-----")
    GPIO.add_event_detect(start_btn_pin, GPIO.RISING, callback=btn_callback, bouncetime=500)
    
    while True:
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(1)
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(1)
finally:
    GPIO.output(led_pin, GPIO.LOW)
    GPIO.cleanup()
    sys.exit(0)