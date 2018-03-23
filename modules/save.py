"""
Upload images to the cloud in a hierarchical time structure.
"""
# coding: utf-8

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os.path
import sys, time, mimetypes
import socket
import cv2
from collections import deque
import threading
import time

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
