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

def save(img, img_time):
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

    Y = img_time.year
    M = img_time.month
    M_str = img_time.strftime('%B')
    D = img_time.day
    h = img_time.hour
    m = img_time.minute
    s = img_time.second
    ms = img_time.microsecond//1000
    hms = str(img_time)[:10] + ' ' + '%02d'%(h) + 'h' + '%02d'%(m) + 'm' + '%02d.%03d'%(s,ms) + 's' + '.jpg'

    if os.path.exists("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D)))):
        cv2.imwrite("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), hms)), img)
    elif os.path.exists("/".join(("detected", str(Y), str(M) + ". " + M_str))):
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D))))
        cv2.imwrite("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), hms)), img)
    elif os.path.exists("/".join(("detected", str(Y)))):
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str)))
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D))))
        cv2.imwrite("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), hms)), img)
    elif os.path.exists("detected"):
        os.mkdir("/".join(("detected", str(Y))))
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str)))
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D))))
        cv2.imwrite("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), hms)), img)
    else:
        os.mkdir("detected")
        os.mkdir("/".join(("detected", str(Y))))
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str)))
        os.mkdir("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D))))
        cv2.imwrite("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), hms)), img)
    
    with open("/".join(("detected", str(Y), str(M) + ". " + M_str, str(D), "activity.log")), "a") as activity_log:
        activity_log.write('%04d/%02d/%02d %02d:%02d:%02d\n' % (Y,M,D,h,m,s))
                                                                                                                                                                                            

def video(time_, fps=30):
    """
    """

    path = "/".join(("detected", str(time_.year), str(time_.month) + ". " + time_.strftime('%B'), str(time_.day)))
    name = ".".join(("detected", str(time_.year), str(time_.month) + ". " + time_.strftime('%B'), str(time_.day)))
    frames = list()

    if os.path.exists(path):

        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.endswith('.avi')]
        files.sort()

        output_path = os.path.join(path, name+'.avi')

        if int(cv2.__version__[0]) < 4:
            video_ = cv2.VideoWriter(output_path+'.tmp.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (480, 640))
        else:
            video_ = cv2.VideoWriter(output_path+'.tmp.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (640, 480))

        if os.path.exists(output_path):
            _video = cv2.VideoCapture(output_path)
            while _video.get(CV_CAP_PROP_POS_FRAMES) < _video.get(CV_CAP_PROP_FRAME_COUNT):
                _, frame = _video.read()
                video_.write(frame)
            _video.release()
            os.remove(output_path)

        for f in files:
            f = os.path.join(path, f)
            frame = cv2.imread(f)
            video_.write(frame)
            # os.remove(f)
        
        video_.release()

        if os.path.exists(output_path+'.tmp.avi'):
            os.rename(output_path+'.tmp.avi', output_path)
