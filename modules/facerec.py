import os
import cv2
import numpy
import collections

from . import imgutils

class FaceRecognizer():

    def __init__(self):
        self.model = None
        self.namedict = dict()
        self.buffer = collections.deque(maxlen=8)
    
    def train(self):
        facedatabase = list()
        for (dirpath, dirnames, filenames) in os.walk('faces'):
            facedatabase.extend(dirnames)
            break
        
        facedatabase = sorted(facedatabase)

        faces, labels = list(), list()
        for i, name in enumerate(facedatabase):
            images = list()
            for (dirpath, dirnames, filenames) in os.walk('faces/'+name):
                images.extend(filenames)
                break
            for image in images:
                faces.append(cv2.cvtColor(cv2.imread('faces/'+name+'/'+image), cv2.COLOR_BGR2GRAY))
                labels.append(i)
            self.namedict[i] = name
        
        if len(facedatabase) > 1:
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model.train(faces, numpy.array(labels))

    def recognize(self, face):
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        self.buffer.append(imgutils.resize(face_gray, 50, 50))
    
        if self.model != None and self.namedict != None and len(self.buffer) == 8:
            detections = list()
            for img in self.buffer:
                label, conf = self.model.predict(img)
                detections.append(label)
            counts = numpy.bincount(detections)
            detectedlabel = numpy.argmax(counts)
            name = self.namedict[detectedlabel] if counts[detectedlabel] > 6 else "Unknown"
            return name