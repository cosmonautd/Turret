""" Module for general conversation
"""

# Standard imports
import os
import datetime

# External imports
import cv2
import dlib
import numpy

# Project imports
import botkit.answer

# Load Cascade Classifiers for upperbody
CASCADE_UPPERBODY = cv2.CascadeClassifier("../resources/cascades/haarcascade_mcs_upperbody.xml")

def detect_pattern(img, cascade, min_rectangle):
    """Pattern detection function.

        Args:
            img: a cv2 image.
            cascade: a CascadeClassifier object.
            min_rectangle: a two element tuple containing width and
                           height of the smaller search window; small
                           values rise the range of vision of our
                           turret, but processing may become slower.

        Returns:
            Coordinates of the rectangle that contains the pattern
            described by the classifier.

        Raises:

    """

    rects = cascade.detectMultiScale(img, 1.2, 3, 1, min_rectangle)

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(coords, img, color=(0,255,0)):
    """Draw a rectangle in an image.

        Args:
            coords: a list of lists. Each sublist has four elements,
                    respectively, top-left and bottom-right, x and y.
                    examples: [[32, 56, 177, 214]]
                              [[32, 56, 177, 214], [44, 53, 194, 217]]
            img: a cv2 image.
            color: a tuple of three elements, the BGR representation
                   of a color for the rectangle.
                   Default is (127, 255, 0).

        Returns:
            The input image, with rectangles placed on the specified
            coordinates.

        Raises:

    """

    for x1, y1, x2, y2 in coords:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

def single_cascade(frame, cascade=CASCADE_UPPERBODY, return_objects=False, drawboxes=True, min_rectangle=(60,60)):
    """
    Use a single cascade to perform object detection
    """

    # Detect cascade pattern in frame
    (rects, frame) = detect_pattern(frame, cascade, min_rectangle)

    # Draw a rectangle around detected patterns if required
    if drawboxes: frame = box(rects, frame)

    # Set found to True if pattern was detected
    found = False
    if len(rects) > 0:
        found = True

    # Return detected pattern coordinates if required + frame and found state
    if return_objects: return frame, found, rects
    else: return frame, found

def im2float(im):
    """
    Convert OpenCV image type to numpy.float
    """
    info = numpy.iinfo(im.dtype)
    return im.astype(numpy.float) / info.max

class Base:
    """
    Basic Teleturret operations
    """
    def __init__(self):
        """
        Initialize answer processor and set callbacks for intents
        """
        self.answer_processor = botkit.answer.AnswerProcessor('base')
        self.answer_processor.set_callback('none', self.none)
        self.answer_processor.set_callback('greetings', self.greetings)
        self.answer_processor.set_callback('someone', self.someone)
        self.answer_processor.set_callback('who', self.who2)
    
    def none(self, message, message_data, answer):
        """
        Postprocess no intent
        Return default response
        """
        return answer

    def greetings(self, message, message_data, answer):
        """
        Postprocess greetings intent
        Return default response
        """
        return answer

    def someone(self, message, message_data, answer):
        """
        Postprocess someone intent
        Get the last detected frame
        Check if lights are turned off or on
        Infer if the room is empty or not based on lights
        """
        # Get path to todays' detections
        now = datetime.datetime.now()
        todaypath = '/'.join(('..', 'detected', str(now.year), str(now.month) + '. ' + now.strftime('%B'), str(now.day)))
        # Get paths for all frames detected today
        detections = list()
        for (_, _, filenames) in os.walk(todaypath):
            detections.extend(filenames)
            break
        detections.sort(reverse=True)
        detections = [d for d in detections if not d.endswith('.avi')]
        # If no detection was made today, infer that nobody went to the lab
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'Nobody here today.'})
        else:
            # If there was a detection today, get the last frame
            lastframepath = os.path.join(todaypath, detections[0])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3: answer.append({'type': 'text', 'text': 'Someone is here!'})
            else: answer.append({'type': 'text', 'text': 'Nobody here.'})
            # Send last detected frame
            answer.append({'type': 'image', 'url': lastframepath})
        return answer

    def who(self, message, message_data, answer):
        """
        Post process who intent
        Infer if there is someone in the room
        If positive, get the last frame in which a face is detected and return
        """
        # Get path to todays' detections
        now = datetime.datetime.now()
        todaypath = '/'.join(('..', 'detected', str(now.year), str(now.month) + '. ' + now.strftime('%B'), str(now.day)))
        # Get paths for all frames detected today
        detections = list()
        for (_, _, filenames) in os.walk(todaypath):
            detections.extend(filenames)
            break
        detections.sort(reverse=True)
        detections = [d for d in detections if not d.endswith('.avi')]
        # If no detection was made today, infer that nobody went to the lab
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'Nobody here today.'})
        else:
            # Load dlib face detector
            detector = dlib.get_frontal_face_detector()
            # If there was a detection today, get the last frame
            lastframepath = os.path.join(todaypath, detections[0])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3:
                answer.append({'type': 'text', 'text': 'Someone is here!'})
                # Check frames from recent to older and try to find a person, skipping 10 by 10
                for i in range(0, len(detections), 10):
                    detection = detections[i]
                    # Get frame
                    framepath = os.path.join(todaypath, detection)
                    frame = cv2.imread(framepath)
                    # Try face detection
                    faces = detector(frame)
                    # If a face was detected, draw a rectangle over it and save, then answer!
                    if len(faces) > 0:
                        for f in faces:
                            cv2.rectangle(frame, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
                        cv2.imwrite('.found.jpg', frame)
                        answer.append({'type': 'text', 'text': 'Target acquired.'})
                        answer.append({'type': 'image', 'url': '.found.jpg'})
                        break
            else:
                answer.append({'type': 'text', 'text': 'Nobody here.'})

        return answer

    def who2(self, message, message_data, answer):
        """
        Post process who intent
        Infer if there is someone in the room
        If positive, get the last frame in which a face is detected and return
        """
        # Get path to todays' detections
        now = datetime.datetime.now()
        todaypath = '/'.join(('..', 'detected', str(now.year), str(now.month) + '. ' + now.strftime('%B'), str(now.day)))
        # Get paths for all frames detected today
        detections = list()
        for (_, _, filenames) in os.walk(todaypath):
            detections.extend(filenames)
            break
        detections.sort(reverse=True)
        detections = [d for d in detections if not d.endswith('.avi')]
        # If no detection was made today, infer that nobody went to the lab
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'Nobody here today.'})
        else:
            # If there was a detection today, get the last frame
            lastframepath = os.path.join(todaypath, detections[0])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3:
                answer.append({'type': 'text', 'text': 'Someone is here!'})
                # Check frames from recent to older and try to find a person, skipping 10 by 10
                for i in range(0, len(detections), 10):
                    detection = detections[i]
                    # Get frame
                    framepath = os.path.join(todaypath, detection)
                    frame = cv2.imread(framepath)
                    # Try upperbody detection
                    # If upperbody was detected, draw a rectangle over it and save, then answer!
                    frame, found = single_cascade(frame, drawboxes=True)
                    if found:
                        cv2.imwrite('.found.jpg', frame)
                        answer.append({'type': 'text', 'text': 'Target acquired.'})
                        answer.append({'type': 'image', 'url': '.found.jpg'})
                        break
            else:
                answer.append({'type': 'text', 'text': 'Nobody here.'})

        return answer

link = Base()
