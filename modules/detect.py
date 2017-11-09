import time
import cv2
from . import imgutils
from . import facerec

detection_modes = [ 'motion',
                    'upperbody-face',
                    'face-recognition']

mode_description = { 'motion': 'Motion detection',
                     'upperbody-face' : 'Upperbody and face detection',
                     'face-recognition' : 'Face detection and recognition' }

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
    frame = imgutils.box(rects_upperbody, frame)
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
def motion_detection(frame, first_frame, thresh=10, it=35, min_area=200, max_area=245760):

    found = False

    raw_frame = frame.copy()

    # Process first_frame
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        found = True

    return frame, raw_frame, found

facerecognizer = None
last = time.time()

def face_recognition(frame):

    global facerecognizer, last

    if not facerecognizer:
        facerecognizer = facerec.FaceRecognizer()
        facerecognizer.train()
    
    if time.time() - last > 10:
        facerecognizer.clear_buffer()

    frame, found, faces = single_cascade(frame, cascade=CASCADE_FACE,
                                                return_faces=True,
                                                drawboxes=True,
                                                min_rectangle=(20,20))
    
    if found:
        x, y, w, h = faces[0]
        face = frame[y:h, x:w]
        name = facerecognizer.recognize(face)
        cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        last = time.time()
    
    return frame, found