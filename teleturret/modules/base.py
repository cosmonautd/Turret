""" Module for general conversation
"""

# Standard imports
import os
import sys
import datetime

# External imports
import cv2
import dlib
import scipy
import numpy
import skimage.exposure
import sklearn.cluster
import matplotlib
import matplotlib.pyplot as plt

# Project imports
import botkit.nlu
import botkit.answer

# Load Cascade Classifiers for upperbody
CASCADE_UPPERBODY = cv2.CascadeClassifier("../resources/cascades/haarcascade_mcs_upperbody.xml")

def log(m):
    print(m)
    sys.stdout.flush()

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
        self.answer_processor.set_callback('who', self.who_activity_graph)
        self.answer_processor.set_callback('activate', self.activate)
        self.answer_processor.set_callback('deactivate', self.deactivate)
        self.answer_processor.set_callback('activity_graph', self.activity_graph)
    
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
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

    def who_face(self, message, message_data, answer):
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
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
    
    def who_upperbody(self, message, message_data, answer):
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
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

    def who_all(self, message, message_data, answer):
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
        else:
            # If there was a detection today, get the last frame
            lastframepath = os.path.join(todaypath, detections[0])
            lastframe = cv2.imread(lastframepath)
            # Check the state of lights
            light = numpy.mean(im2float(lastframe).flatten())
            # Infer if there is someone in the lab
            if light > 0.3:
                people = list()
                answer.append({'type': 'text', 'text': 'Someone is here!'})
                # Check frames from recent to older and try to find a person, skipping 10 by 10
                for i in range(0, len(detections), 10):
                    detection = detections[i]
                    # Get frame
                    framepath = os.path.join(todaypath, detection)
                    frame = cv2.imread(framepath)
                    # Try upperbody detection
                    # If upperbody was detected, draw a rectangle over it and save, then answer!
                    frame, found, rects = single_cascade(frame, drawboxes=False, return_objects=True)
                    if found: people.append((frame, rects))
                    if len(people) > 15: break
                # Clustering
                features = list()
                for frame, rects in people:
                    x, y, w, h = rects[0]
                    crop = frame[y:h, x:w].astype(numpy.float32)
                    crop_ch0 = crop[:,:,0]
                    crop_ch1 = crop[:,:,1]
                    crop_ch2 = crop[:,:,2]
                    hist0 = skimage.exposure.histogram(crop_ch0, nbins=12, normalize=True)[0]
                    hist1 = skimage.exposure.histogram(crop_ch1, nbins=12, normalize=True)[0]
                    hist2 = skimage.exposure.histogram(crop_ch2, nbins=12, normalize=True)[0]
                    log(hist0.shape)
                    log(hist1.shape)
                    log(hist2.shape)
                    ft = numpy.concatenate((hist0, hist1, hist2), axis=None)
                    features.append(ft)
                features = numpy.array(features)
                log(features.shape)
                clustering = sklearn.cluster.AffinityPropagation()
                clustering.fit(features)
                labels = clustering.labels_
                n_labels = len(numpy.unique(labels))
                # Selecting
                selected_labels = list()
                selected_frames = list()
                for i, p in enumerate(people):
                    if labels[i] not in selected_labels:
                        selected_frames.append(people[i][0])
                        selected_labels.append(labels[i])
                answer.append({'type': 'text', 'text': 'Targets acquired.'})
                for i, f in enumerate(selected_frames):
                    cv2.imwrite('.found-%02d.jpg' % (i), f)
                    answer.append({'type': 'image', 'url': '.found-%02d.jpg' % (i)})
                answer.append({'type': 'text', 'text': 'This was a test using affinity propagation clustering.'})
            else:
                answer.append({'type': 'text', 'text': 'Nobody here.'})

        return answer
    
    def who_activity_graph(self, message, message_data, answer):
        """
        Post process who intent
        Infer if there is someone in the room
        If positive, return last five events
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
        else:
            message = ''
            counts = numpy.zeros(24*60*60)
            # Check frames from recent to older and accumulate counts
            for i in range(0, len(detections)):
                t = detections[i]
                h = int(t.split()[1].split('h')[0])
                m = int(t.split()[1].split('h')[1].split('m')[0])
                s = int(t.split()[1].split('h')[1].split('m')[1].split('.')[0])
                counts[3600*h + 60*m + s] += 1
            xaxis = numpy.arange(0, len(counts))
            # Generate graph
            fig, ax = plt.subplots()
            # Identify peaks
            peaks, _ = scipy.signal.find_peaks(counts, height=8, distance=6)
            ax.plot(xaxis, counts)
            ax.plot(peaks, counts[peaks], "x")
            ax.set(xlabel='Time', ylabel='Detections', title='Activity Graph')
            formatter = matplotlib.ticker.FuncFormatter(lambda s, x: '%02d:%02d' % (s//3600,(s%3600)//60))
            ax.xaxis.set_major_formatter(formatter)
            fig.savefig(".activity.png", dpi=300, bbox_inches='tight')
            if len(peaks) > 0:
                # Iterate over the peaks
                answer.append({'type': 'text', 'text': 'Targets acquired.'})
                selected_frames = list()
                for peak in peaks[::-1][:5]:
                    t = '%02dh%02dm%02d' % (peak//3600,(peak%3600)//60, peak%60)
                    for i in range(0, len(detections)):
                        if detections[i].split()[1].startswith(t):
                            # Selecting
                            j = i + 6
                            framepath = os.path.join(todaypath, detections[j])
                            frame = cv2.imread(framepath)
                            selected_frames.append(frame)
                            break
                for i, f in enumerate(selected_frames):
                    cv2.imwrite('.found-%02d.jpg' % (i), f)
                    answer.append({'type': 'image', 'url': '.found-%02d.jpg' % (i)})
            else:
                answer.append({'type': 'text', 'text': 'Oops, there was no significant activity today.'})

        return answer
    
    def activate(self, message, message_data, answer):
        """
        Post process activate intent
        Infer what needs to be activated
        """
        feature = ''
        if 'text_en' in message_data and 'notification' in message_data['text_en'] \
            or 'text' in message_data and 'notification' in message_data['text']:
            feature = 'notifications'

        if feature != '':
            context = botkit.nlu.Context()
            context.write(message['username'], feature, True)
            answer.append({'type': 'text', 'text': '%s activated' % (feature.capitalize())})
        else:
            answer.append({'type': 'text', 'text': 'Unknown error...'})

        return answer
    
    def deactivate(self, message, message_data, answer):
        """
        Post process deactivate intent
        Infer what needs to be deactivated
        """
        feature = ''
        if 'text_en' in message_data and 'notification' in message_data['text_en'] \
            or 'text' in message_data and 'notification' in message_data['text']:
            feature = 'notifications'

        if feature != '':
            context = botkit.nlu.Context()
            context.write(message['username'], feature, False)
            answer.append({'type': 'text', 'text': '%s deactivated' % (feature.capitalize())})
        else:
            answer.append({'type': 'text', 'text': 'Unknown error...'})

        return answer
    
    def activity_graph(self, message, message_data, answer):
        """
        Post process activity_graph intent
        Generates daily activity graph
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
            answer.append({'type': 'text', 'text': 'Nobody was here today.'})
        else:
            message = ''
            counts = numpy.zeros(24*60*60)
            # Check frames from recent to older and accumulate counts
            for i in range(0, len(detections)):
                t = detections[i]
                h = int(t.split()[1].split('h')[0])
                m = int(t.split()[1].split('h')[1].split('m')[0])
                s = int(t.split()[1].split('h')[1].split('m')[1].split('.')[0])
                counts[3600*h + 60*m + s] += 1
            offset = 3600*h + 60*m + s
            counts = numpy.trim_zeros(counts)
            xaxis = numpy.arange(offset, offset + len(counts))
            # Generate graph
            fig, ax = plt.subplots()
            ax.plot(xaxis, counts)
            ax.set(xlabel='Time', ylabel='Detections', title='Activity Graph')
            formatter = matplotlib.ticker.FuncFormatter(lambda s, x: '%02d:%02d' % (s//3600,(s%3600)//60))
            ax.xaxis.set_major_formatter(formatter)
            fig.savefig(".activity.png", dpi=300, bbox_inches='tight')
            # Answer with graph
            answer.append({'type': 'text', 'text': 'Sending you today\'s activity graph...'})
            answer.append({'type': 'image', 'url': '.activity.png'})

        return answer

link = Base()
