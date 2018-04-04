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
        self.answer_processor.set_callback('greetings', self.greetings)
        self.answer_processor.set_callback('someone', self.someone)
        self.answer_processor.set_callback('who', self.who)

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
        # If no detection was made today, infer that nobody went to the lab
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'I think nobody went to the lab today'})
        else:
            # If there was a detection today, get the last frame
            detections.sort()
            lastframepath = os.path.join(todaypath, detections[-1])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3: answer.append({'type': 'text', 'text': 'Someone is in the lab!'})
            else: answer.append({'type': 'text', 'text': 'I think the lab is empty'})
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
        # If no detection was made today, infer that nobody went to the lab
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'I think nobody went to the lab today'})
        else:
            # Load dlib face detector
            detector = dlib.get_frontal_face_detector()
            # If there was a detection today, get the last frame
            detections.sort(reverse=True)
            lastframepath = os.path.join(todaypath, detections[0])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3:
                answer.append({'type': 'text', 'text': 'Someone is in the lab!'})
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
                        cv2.imwrite('.face.jpg', frame)
                        answer.append({'type': 'text', 'text': 'Just found this test subject'})
                        answer.append({'type': 'image', 'url': '.face.jpg'})
                        break
            else:
                answer.append({'type': 'text', 'text': 'I think the lab is empty'})
            
        return answer

link = Base()
