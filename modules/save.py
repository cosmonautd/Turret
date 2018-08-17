"""
Upload images to the cloud in a hierarchical time structure.
"""
# coding: utf-8

# Standard imports
import os
import sys
import time
import socket
import mimetypes
import threading
from collections import deque

# External imports
import cv2

CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_FRAME_COUNT = 7

def save(img, img_time, uploadqueue=None):
    """Save images to disc or a Google Drive account.

        Save an image in a hierarchical structure inside the detected/
        folder -> year/month/day/image Additionally saves the image
        using a similar structure inside a Google Drive account, if set.

        Args:
            img: a cv2 image.
            img_time: the time of capture.
            google: a Drive object.

        Returns:

        Raises:

    """

    if os.path.exists("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day)))):
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:23] + "s" + ".png")), img) 
    elif os.path.exists("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B')))):
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:23] + "s" + ".png")), img) 
    elif os.path.exists("/".join(("detected", str(img_time.year)))):
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:23] + "s" + ".png")), img) 
    elif os.path.exists("detected"):
        os.mkdir("/".join(("detected", str(img_time.year))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:23] + "s" + ".png")), img) 
    else:
        os.mkdir("detected")
        os.mkdir("/".join(("detected", str(img_time.year))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:23] + "s" + ".png")), img) 

    if uploadqueue:
        uploadqueue.append(img_time) 

def video(time_, fps=30):
    """
    """

    path = "/".join(("detected", str(time_.year), str(time_.month) + ". " + time_.strftime('%B'), str(time_.day)))
    name = ".".join(("detected", str(time_.year), str(time_.month) + ". " + time_.strftime('%B'), str(time_.day)))
    frames = list()
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.endswith('.avi')]
    files.sort()

    output_path = os.path.join(path, name+'.avi')

    if os.path.exists(output_path):
        _video = cv2.VideoCapture(output_path)
        while _video.get(CV_CAP_PROP_POS_FRAMES) < _video.get(CV_CAP_PROP_FRAME_COUNT):
            _, frame = _video.read()
            frames.append(frame)
        _video.release()

    for f in files:
        f = os.path.join(path, f)
        frame = cv2.imread(f)
        os.remove(f)
        h, w, d = frame.shape
        size = (w, h)
        frames.append(frame)
    
    if len(frames) > 0:
        video_ = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        for frame in frames:
            video_.write(frame)
        video_.release()