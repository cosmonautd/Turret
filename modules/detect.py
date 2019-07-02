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
                        'face-recognition',
                        'gesture-recognition']

    mode_description = {    'motion': 'Motion detection',
                            'upperbody-face' : 'Upperbody and face detection',
                            'face-recognition' : 'Face detection and recognition',
                            'gesture-recognition' : 'Gesture recognition' }

# Load Cascade Classifiers for upperbody and face
CASCADE_UPPERBODY = cv2.CascadeClassifier("resources/cascades/haarcascade_mcs_upperbody.xml")
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


train_counter = 0
X = []
Y = []
trained = False
classifier = None
roi_buffer = collections.deque(maxlen=1)
def gesture_recognition(frame):
    """ Perform gesture recognition
    """
    global motion_detection_buffer

    min_area = 150*150
    found = False

    # Draw a rectangle around hand detection area (Requires 640x480 frame)
    cv2.rectangle(frame, (50-2, 50-2), (350+2, 350+2), (0,255,0), 2)

    crop = frame[50:350,50:350]
    raw_crop = crop.copy()
    show_crop = crop.copy()

    if len(motion_detection_buffer) > 0:

        global train_counter, X, Y, trained, classifier
        if train_counter < 50:
            # Process first_crop
            first_crop = cv2.cvtColor(motion_detection_buffer[-1], cv2.COLOR_BGR2GRAY)
            first_crop = cv2.GaussianBlur(first_crop, (21, 21), 0)

            # Resize the frame, convert it to grayscale, and blur it
            # crop = imgutils.resize(crop, width=100)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.GaussianBlur(crop, (21, 21), 0)

            # Compute the absolute difference between the current frame and first frame
            delta = cv2.absdiff(first_crop, crop)
            delta = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))
            delta = cv2.morphologyEx(delta, cv2.MORPH_CLOSE, kernel)

            frame[50:350,50:350] = cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB)
            X = [p for row in raw_crop for p in row]
            Y = [1 if p > 100 else 0 for row in delta for p in row]
            train_counter += 1
        elif not trained:
            from sklearn.neural_network import MLPClassifier
            classifier = MLPClassifier(hidden_layer_sizes=(5, 10, 5), activation='logistic', solver='adam', learning_rate='adaptive')
            classifier.fit(X, Y)
            trained = True
        elif trained:
            segm = classifier.predict(raw_crop.reshape(300*300, 3))
            segm = segm*255
            segm = segm.reshape(300,300,1)
            segm = segm.astype(numpy.float32)
            frame[50:350,50:350] = cv2.cvtColor(segm, cv2.COLOR_GRAY2RGB)
        
        # # Dilate the thresholded image to fill in holes, then find contours on thresholded image
        # dilated = cv2.dilate(delta, None, iterations=15)
        # (_, cnts, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE)
        
        # if len(cnts) == 0:
        #     try:
        #         cv2.rectangle(show_crop, roi_buffer[-1]['top-left'], roi_buffer[-1]['bottom-right'], (0, 0, 255), 2)
        #     except:
        #         pass
        # else:
        #     max_cnt = cnts[0]
        #     for c in cnts:
        #         if (len(c) > len(max_cnt)):
        #             max_cnt = c
        #     (x, y, w, h) = cv2.boundingRect(max_cnt)
        #     if w*h > min_area:
        #         roi_buffer.append({'x': x, 'y': y, 'w': w, 'h': h, 'top-left': (x, y), 'bottom-right':(x + w, y + h)})
        #     try:
        #         cv2.rectangle(show_crop, roi_buffer[-1]['top-left'], roi_buffer[-1]['bottom-right'], (0, 0, 255), 2)
        #     except:
        #         pass
            
        # frame[50:350,50:350] = show_crop
    
    if train_counter == 0:
        motion_detection_buffer.append(raw_crop)

    # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # crop = cv2.medianBlur(crop, 5)
    # (value, crop) = cv2.threshold(crop, 180, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
    # # frame[50:350,50:350] = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    # _, contours, _ = cv2.findContours(crop, 1, 2)
    # if len(contours) == 0:
    #     return frame, False
    # cnt = contours[0]
    # for c in contours:
    #     if (len(c) > len(cnt)):
    #         cnt = c
    # approx = cv2.approxPolyDP(cnt, epsilon = 2, closed = True)
    # x, y, w, h = cv2.boundingRect(approx)
    # cv2.rectangle(crop, (x,y), (x+w,y+h), (255,255,255), 2)
    # hand = crop[y:(y+h),x:(x+w)]
    # hand = cv2.resize(hand, (100,100), interpolation = cv2.INTER_AREA)
    
    return frame, False