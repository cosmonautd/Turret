""" Module for general conversation
"""

import os
import datetime

import cv2
import numpy

import botkit.answer

def im2double(im):
    info = numpy.iinfo(im.dtype)
    return im.astype(numpy.float) / info.max

class Base:
    """
    """
    def __init__(self):
        self.answer_processor = botkit.answer.AnswerProcessor('base')
        self.answer_processor.set_callback('greetings', self.greetings)
        self.answer_processor.set_callback('someone', self.someone)

    def greetings(self, message, message_data, answer):
        """ Postprocess greetings
        """
        return answer

    def someone(self, message, message_data, answer):
        """ Postprocess someone
        """
        now = datetime.datetime.now()
        todaypath = '/'.join(('..', 'detected', str(now.year), str(now.month) + '. ' + now.strftime('%B'), str(now.day)))
        detections = list()
        for (_, _, filenames) in os.walk(todaypath):
            detections.extend(filenames)
            break
        if len(detections) == 0:
            answer.append({'type': 'text', 'text': 'I think nobody went to the lab today'})
        else:
            detections.sort()
            lastframepath = os.path.join(todaypath, detections[-1])
            lastframe = cv2.imread(lastframepath)
            light = numpy.mean(im2double(lastframe).flatten())
            if light > 0.3: answer.append({'type': 'text', 'text': 'Someone is in the lab!'})
            else: answer.append({'type': 'text', 'text': 'I think the lab is empty'})
            answer.append({'type': 'image', 'url': lastframepath})
        return answer

link = Base()
