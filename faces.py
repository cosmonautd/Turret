import os
import cv2
import sys
import numpy
import shutil
import signal
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GObject, GLib, Gio

from modules import detect
from modules import imgutils

# Width and height of the frames Face Manager will process
WIDTH  = 320
HEIGHT = 240
# Codes for OpenCV camera settings
CV_CAP_PROP_FRAME_WIDTH  = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

class FaceManager:
    """
    A class to control Face Manager GUI operations.
    """

    def __init__(self):
        """
        FaceManager constructor

        Load GUI template from face_manager.glade into a Gtk.Builder object.
        Intantiate objects for each necessary GUI element.
        """
        self.gladefile = 'resources/gui/faces.glade'
        self.gtk = Gtk.Builder()
        self.gtk.add_from_file(self.gladefile)
        self.gtk.connect_signals(self)

        self.FaceFrame = self.gtk.get_object("FaceFrame")
        self.NewFaceName = self.gtk.get_object("NewFaceName")
        self.NewFaceButton = self.gtk.get_object("NewFaceButton")
        self.DeleteFaceButton = self.gtk.get_object("DeleteFaceButton")
        self.FaceManagerDatabase = self.gtk.get_object("DatabaseGrid")

        self.camera = None
        self.face_recognizer = None
        self.namedict = None
        self.last_images = list()

        self.init_newface_name()
        self.init_newface_button()
        self.init_deleteface_button()
        self.update_face_manager_database()

        self.MainWindow = self.gtk.get_object("MainWindow")
        self.MainWindow.connect("delete-event", self.close_button_pressed)
        self.MainWindow.show_all()

        self.name = ""
        self.face_samples_recorded = 0
        self.face_samples = list()
        self.get_face_samples = False

        self.init_camera()
        GLib.idle_add(self.camera_preview)
    
    def update_face_manager_database(self):
        """
        """
        facedatabase = list()
        for (dirpath, dirnames, filenames) in os.walk('faces'):
            facedatabase.extend(dirnames)
            break
        
        facedatabase = sorted(facedatabase)
        
        for child in self.FaceManagerDatabase.get_children():
            self.FaceManagerDatabase.remove(child)
        
        for i, name in enumerate(facedatabase):
            button = Gtk.ToggleButton()
            box = Gtk.Box()
            image = Gtk.Image()
            image.set_from_file('faces/' + name + '/0.jpg')
            label = Gtk.Label(name)
            box.add(image)
            box.pack_start(label, False, True, 15)
            button.add(box)
            button.set_name(name)
            self.FaceManagerDatabase.add(button)
        self.FaceManagerDatabase.show_all()
        self.trainfacerec()

    def init_newface_name(self):
        """
        Connect the method changed_newface_name() to NewFaceName.
        """
        self.NewFaceName.connect("changed", self.changed_newface_name)

    def changed_newface_name(self, entry):
        """
        Verify if the input name is valid.
        """
        name = self.NewFaceName.get_text()
        if all(c.isalpha() or c.isspace() for c in name):
            self.name = name
            self.NewFaceButton.set_sensitive(True)
        else:
            self.NewFaceButton.set_sensitive(False)

    def init_newface_button(self):
        """
        Connect the method clicked_newface_button() to NewFaceButton.
        """
        self.NewFaceButton.connect("clicked", self.clicked_newface_button)

    def clicked_newface_button(self, button):
        """
        Start new face acquisition process.
        """
        self.start_face_samples_acquisition()
    
    def init_deleteface_button(self):
        """
        Connect the method clicked_deleteface_button() to DeleteFaceButton.
        """
        self.DeleteFaceButton.connect("clicked", self.clicked_deleteface_button)

    def clicked_deleteface_button(self, button):
        """
        Delete all selected faces from dataset.
        """
        self.delete_face_samples()
    
    def clicked_deleteface_button(self, button):
        """
        Start new face acquisition process.
        """
        self.delete_face_samples()
    
    def delete_face_samples(self):
        """
        """
        for child in self.FaceManagerDatabase.get_children():
            if child.get_active():
                shutil.rmtree('faces/' + child.get_name())
                self.FaceManagerDatabase.remove(child)

    def start_face_samples_acquisition(self):
        """
        """
        self.face_samples_recorded = 0
        self.NewFaceName.set_sensitive(False)
        self.NewFaceButton.set_sensitive(False)
        self.get_face_samples = True
        

    def init_camera(self):
        """
        Initiate video capture using the first webcam found.
        Set camera width and height settings.
        """
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH)
        self.camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    def camera_preview(self):
        """
        Get a new frame from camera.
        Process this frame.
        Write to an image file, which is awful, but the only way it worked.
        Display the file on the Frame element of the main window.
        """
        retval, frame = self.camera.read()
        frame = imgutils.crop(frame, 70, 320 - 70, 0, 240)

        frame, found, faces = detect.single_cascade(frame, cascade=detect.CASCADE_FACE,
                                                           return_faces=True,
                                                           drawboxes=True)

        if found:

            x, y, w, h = faces[0]
            face_gray = cv2.cvtColor(frame[y:h, x:w], cv2.COLOR_BGR2GRAY)
            self.last_images.append(imgutils.resize(face_gray, 50, 50))
        
            if self.face_recognizer != None and self.namedict != None and len(self.last_images) > 8:
                detections = list()
                for img in self.last_images[-8:]:
                    label, conf = self.face_recognizer.predict(img)
                    detections.append(label)
                counts = numpy.bincount(detections)
                detectedlabel = numpy.argmax(counts)
                name = self.namedict[detectedlabel] if counts[detectedlabel] > 6 else "Unknown"
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.imwrite(".frame.jpg", frame)
        # pixbuf_frame = GdkPixbuf.Pixbuf.new_from_file(".frame.jpg")
        # self.FaceFrame.set_from_pixbuf(pixbuf_frame)

        h, w, d = frame.shape
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(frame_show.flatten(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*3, None, None)
        self.FaceFrame.set_from_pixbuf(pixbuf)

        if self.get_face_samples:

            if found:
                self.face_samples_recorded += 1
                x, y, w, h = faces[0]
                self.face_samples.append(frame[y:h, x:w])

            if self.face_samples_recorded+1 > 50:
                self.save_face_samples()

        return True

    def save_face_samples(self):
        """
        Save captured face samples to our face database
        """
        outputpath = 'faces/' + self.name
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        for i, face in enumerate(self.face_samples):
            cv2.imwrite(outputpath + '/' + str(i+1) + '.jpg', imgutils.resize(face, 50, 50))
        cv2.imwrite(outputpath + '/' + str(0) + '.jpg', imgutils.resize(self.face_samples[0], 50, 50))
        self.clean_after_acquisition()

    def clean_after_acquisition(self):
        """
        Make cleaning after adding a new face
        """
        self.get_face_samples = False
        self.face_samples_recorded = 0
        self.face_samples = list()
        self.NewFaceName.set_text("")
        self.NewFaceName.set_sensitive(True)
        self.NewFaceButton.set_sensitive(True)
        self.update_face_manager_database()
    
    def trainfacerec(self):
        facedatabase = list()
        for (dirpath, dirnames, filenames) in os.walk('faces'):
            facedatabase.extend(dirnames)
            break
        
        facedatabase = sorted(facedatabase)

        self.namedict = dict()
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
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train(faces, numpy.array(labels))


    def close_button_pressed(self, widget, event):
        """
        If close button is pressed, better clean our resources.
        Turn off the camera.
        Execute clean() method, for other resources deallocations.
        Then, actually shut down the poor turret with Gtk.main_quit().
        """
        if self.camera:
            self.camera.release()
        clean()
        Gtk.main_quit()

def clean():
    """
    Use this to close the turret's modules when shutting down.
    """
    pass

def sigint_handler(signum, instant):
    """
    Capture SIGINT signal and quit safely.
    """
    clean()
    sys.exit()

if __name__ == "__main__":

    # Activate capture of SIGINT (Ctrl-C)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        GObject.threads_init()
        f = FaceManager()
        Gtk.main()
    except KeyboardInterrupt:
        pass
