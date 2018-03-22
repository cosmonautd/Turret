import os
import sys
import time
import collections
import cv2
import numpy
from . import imgutils
from . import facerec

# Set locale (standardize month names)
if sys.platform == "linux" or sys.platform == "linux2":

    import face_recognition as fc

    detection_modes = [ 'motion',
                        'upperbody-face',
                        'face-recognition']

    mode_description = {    'motion': 'Motion detection',
                            'upperbody-face' : 'Upperbody and face detection',
                            'face-recognition' : 'Face detection and recognition' }

elif sys.platform == "win32":

    detection_modes = [ 'motion',
                        'upperbody-face']

    mode_description = {    'motion': 'Motion detection',
                            'upperbody-face' : 'Upperbody and face detection' }

# Load Cascade Classifiers for upperbody and face
# We use classifiers commonly found in opencv packages
CASCADE_UPPERBODY = cv2.CascadeClassifier("resources/cascades/haarcascade_mcs_upperbody.xml")
# CASCADE_FACE = cv2.CascadeClassifier("resources/cascades/haarcascade_frontalface_alt.xml")
CASCADE_FACE = cv2.CascadeClassifier("resources/cascades/lbpcascade_frontalface_improved.xml")
CASCADE_PROFILE_FACE = cv2.CascadeClassifier("resources/cascades/haarcascade_profileface.xml")

def single_cascade(frame, cascade=CASCADE_UPPERBODY, return_faces=False, drawboxes=True, min_rectangle=(60,60)):

    # Detect cascade pattern in the frame and draw a green rectangle around it,
    # if pattern is found.
    (rects, frame) = imgutils.detect_pattern(frame, cascade, min_rectangle)

    if drawboxes:
        frame = imgutils.box(rects, frame)

    found = False
    if len(rects) > 0:
        found = True

    if return_faces:
        return frame, found, rects
    else:
        return frame, found

def double_cascade(frame, return_faces=False,
                    cascade_upperbody=CASCADE_UPPERBODY, cascade_face=CASCADE_FACE):

    # Detect upperbodies in the frame and draw a green rectangle around it, if found
    (rects_upperbody, frame) = imgutils.detect_pattern(frame, cascade_upperbody, (60,60))
    frame = imgutils.box(rects_upperbody, frame, (0, 0, 255))
    rects_face = []
    found = False
    # Search for upperbodies!
    if len(rects_upperbody) > 0:

        # For each upperbody detected, search for faces! (Removes false positives)
        for x, y, w, h in rects_upperbody:
            frame_crop = frame[y:h, x:w]
            (rects_face, frame_crop) = imgutils.detect_pattern(frame_crop, cascade_face, (25,25))

            # For each face detected, make some drawings around it
            for xf, yf, wf, hf in rects_face:

                xf += x
                yf += y
                wf += x
                hf += y

                #cv2.circle(frame, ((w+x)/2, (h+y)/2), 10, (255,0,0), thickness=1, lineType=8, shift=0)
                #cv2.circle(frame, (wf, hf), 10, (0,0,255), thickness=1, lineType=8, shift=0)

                frame = imgutils.box([[xf, yf, wf, hf]], frame, (0, 0, 255))

    if len(rects_face) > 0:
        found = True

    if return_faces:
        return frame, found, [ (xf+x, yf+y, wf+x, hf+y) for xf, yf, wf, hf in rects_face for x, y, w, h in rects_upperbody]
    else:
        return frame, found

# based on a tutorial from http://www.pyimagesearch.com/
motion_detection_buffer = collections.deque(maxlen=1)
def motion_detection(frame, thresh=10, it=35, min_area=200, max_area=numpy.inf, drawboxes=True):

    global motion_detection_buffer

    found = False
    raw_frame = frame.copy()

    if len(motion_detection_buffer) > 0:

        # Process first_frame
        first_frame = cv2.cvtColor(motion_detection_buffer[-1], cv2.COLOR_BGR2GRAY)
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

        # resize the frame, convert it to grayscale, and blur it
        #frame = imgutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frameDelta, thresh, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=it)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue

            # compute the bounding box for the contour, draw it on the frame and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if w*h > max_area: continue
            if drawboxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            found = True
    
    motion_detection_buffer.append(raw_frame)

    return frame, found

facerecognizer = None
last = time.time()

def old_face_recognition(frame):

    global facerecognizer, last

    if not facerecognizer:
        facerecognizer = facerec.FaceRecognizer(buffersize=10)
        facerecognizer.train()
    
    if time.time() - last > 10:
        facerecognizer.clear_buffer()

    frame, found, faces = single_cascade(frame, cascade=CASCADE_FACE,
                                                return_faces=True,
                                                drawboxes=True,
                                                min_rectangle=(40,40))
    
    if found:
        x, y, w, h = faces[0]
        face = frame[y:h, x:w]
        name = facerecognizer.recognize(face)
        cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        last = time.time()
    
    return frame, found

database = None
facedatabase = None
facedatabase_encodings = None
fraction = 0.25

def face_recognition(frame):

    global database, facedatabase, facedatabase_encodings, fraction

    found = False

    if (not database) or (not facedatabase) or (not facedatabase_encodings):
        database = list()
        for (dirpath, dirnames, filenames) in os.walk('faces'):
            database.extend(filenames)
            break
        facedatabase = [fc.load_image_file(os.path.join('faces', name)) for name in database]
        facedatabase_encodings = [fc.face_encodings(face)[0] for face in facedatabase]
    
    small_frame = cv2.resize(frame, (0, 0), fx=fraction, fy=fraction)
    face_locations = fc.face_locations(small_frame)
    face_encodings = fc.face_encodings(small_frame, face_locations)

    if len(face_encodings) > 0:

        found = True

        face_names = []
        for face_encoding in face_encodings:
            match = fc.compare_faces(facedatabase_encodings, face_encoding, tolerance=0.5)
            try: name = database[match.index(True)].split('.')[0]
            except ValueError: name = "Unknown"
            face_names.append(name)
        
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
    
    return frame, found