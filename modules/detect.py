""" Module that defines detection modes
"""

# Standard imports
import os
import sys
import time
import collections

# External imports
import cv2
import numpy

# Project imports
from . import imgutils

# Support for Linux only
if sys.platform == "linux" or sys.platform == "linux2":

    import face_recognition as fc

    detection_modes = [ 'motion',
                        'upperbody-face',
                        'face-recognition']

    mode_description = {    'motion': 'Motion detection',
                            'upperbody-face' : 'Upperbody and face detection',
                            'face-recognition' : 'Face detection and recognition' }

# Load Cascade Classifiers for upperbody and face
CASCADE_UPPERBODY = cv2.CascadeClassifier("resources/cascades/haarcascade_upperbody.xml")
CASCADE_FACE = cv2.CascadeClassifier("resources/cascades/lbpcascade_frontalface_improved.xml")
CASCADE_PROFILE_FACE = cv2.CascadeClassifier("resources/cascades/haarcascade_profileface.xml")

def single_cascade(frame, cascade=CASCADE_UPPERBODY, return_objects=False, drawboxes=True, min_rectangle=(60,60)):
    """
    Use a single cascade to perform object detection
    """

    # Detect cascade pattern in frame
    (rects, frame) = imgutils.detect_pattern(frame, cascade, min_rectangle)

    # Draw a rectangle around detected patterns if required
    if drawboxes: frame = imgutils.box(rects, frame)

    # Set found to True if pattern was detected
    found = False
    if len(rects) > 0:
        found = True

    # Return detected pattern coordinates if required + frame and found state
    if return_objects: return frame, found, rects
    else: return frame, found

def double_cascade(frame, first_cascade=CASCADE_UPPERBODY, second_cascade=CASCADE_FACE, return_objects=False, drawboxes=True):
    """
    Use two cascades to perform object detection
    """

    # Detect upperbodies in the frame
    (rects_first_cascade, frame) = imgutils.detect_pattern(frame, first_cascade, (60,60))

    # Draw a rectangle around detected patterns if required
    if drawboxes: frame = imgutils.box(rects_first_cascade, frame, (0, 0, 255))
    
    # Prepare face detection inside detected upperbodies
    rects_second_cascade = []
    if len(rects_first_cascade) > 0:

        # For each upperbody detected, search for faces
        for x, y, w, h in rects_first_cascade:
            frame_crop = frame[y:h, x:w]
            (rects_second_cascade, frame_crop) = imgutils.detect_pattern(frame_crop, second_cascade, (25,25))

            # For each face detected, draw a rectangle if required
            if drawboxes:
                for xf, yf, wf, hf in rects_second_cascade:

                    xf += x
                    yf += y
                    wf += x
                    hf += y

                    #cv2.circle(frame, ((w+x)/2, (h+y)/2), 10, (255,0,0), thickness=1, lineType=8, shift=0)
                    #cv2.circle(frame, (wf, hf), 10, (0,0,255), thickness=1, lineType=8, shift=0)

                    frame = imgutils.box([[xf, yf, wf, hf]], frame, (0, 0, 255))

    # Set found to True if a face was detected
    found = False
    if len(rects_second_cascade) > 0:
        found = True

    # Return detected face coordinates if required + frame and found state
    if return_objects:
        return frame, found, [ (xf+x, yf+y, wf+x, hf+y) for xf, yf, wf, hf in rects_second_cascade for x, y, w, h in rects_first_cascade]
    else:
        return frame, found

# Based on a tutorial from http://www.pyimagesearch.com/
motion_detection_buffer = collections.deque(maxlen=1)
def motion_detection(frame, thresh=10, it=35, min_area=200, max_area=numpy.inf, drawboxes=True):
    """ Detect if significant motion happened between two frames
    """
    global motion_detection_buffer

    found = False
    raw_frame = frame.copy()

    if len(motion_detection_buffer) > 0:

        # Process first_frame
        first_frame = cv2.cvtColor(motion_detection_buffer[-1], cv2.COLOR_BGR2GRAY)
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

        # Resize the frame, convert it to grayscale, and blur it
        #frame = imgutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frameDelta, thresh, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=it)
        if int(cv2.__version__[0]) < 4:
            _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for c in cnts:
            # If the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue

            # Compute the bounding box for the contour, draw it on the frame and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h > max_area: continue
            if drawboxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            found = True
    
    motion_detection_buffer.append(raw_frame)

    return frame, found

# Variables required for face recognition function
database = None
facedatabase = None
facedatabase_encodings = None
fraction = 0.25

def face_recognition(frame, drawboxes=True):
    """ Perform face recognition using face_recognition package
    """
    global database, facedatabase, facedatabase_encodings, fraction

    # Define standard found state
    found = False

    # Initialize face database if not already initialized
    if (not database) or (not facedatabase) or (not facedatabase_encodings):
        database = list()
        # Search for known faces in faces/ directory
        for (_, _, filenames) in os.walk('faces'):
            database.extend(filenames)
            break
        # Populate face database and generate face encodings
        facedatabase = [fc.load_image_file(os.path.join('faces', name)) for name in database]
        facedatabase_encodings = [fc.face_encodings(face)[0] for face in facedatabase]
    
    # Create a resized copy of the frame in order to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=fraction, fy=fraction)

    # Detect faces and generate their encodings
    face_locations = fc.face_locations(small_frame)
    face_encodings = fc.face_encodings(small_frame, face_locations)

    # Recognize faces if found
    if len(face_encodings) > 0:

        found = True

        # Recognize faces and determine their names
        face_names = []
        for face_encoding in face_encodings:
            match = fc.compare_faces(facedatabase_encodings, face_encoding, tolerance=0.5)
            try: name = database[match.index(True)].split('.')[0]
            except ValueError: name = "Unknown"
            face_names.append(name)
        
        # Draw a rectangle and name around recognized faces if required
        if drawboxes:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name != "Unknown":
                    top = int((1/fraction)*top - 16)
                    right = int((1/fraction)*right + 16)
                    bottom = int((1/fraction)*bottom + 16)
                    left = int((1/fraction)*left - 16)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left-1, top - 20), (max(right+1, left+12*len(name)), top), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)
    
    # Return frame and found state
    return frame, found