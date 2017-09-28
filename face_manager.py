import os
import cv2
import sys
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
        self.gladefile = 'resources/gui/face_manager.glade'
        self.gtk = Gtk.Builder()
        self.gtk.add_from_file(self.gladefile)
        self.gtk.connect_signals(self)

        self.FaceFrame = self.gtk.get_object("FaceFrame")
        self.NewFaceName = self.gtk.get_object("NewFaceName")
        self.NewFaceButton = self.gtk.get_object("NewFaceButton")
        self.DeleteFaceButton = self.gtk.get_object("DeleteFaceButton")
        self.FaceManagerDatabase = self.gtk.get_object("DatabaseGrid")

        self.camera = None
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
    
    def update_face_manager_database(self):
        """
        """
        facedatabase = list()
        for (dirpath, dirnames, filenames) in os.walk('faces'):
            facedatabase.extend(dirnames)
            break
        
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
        self.init_camera()
        self.NewFaceName.set_sensitive(False)
        self.NewFaceButton.set_sensitive(False)
        GLib.idle_add(self.get_face_samples)

    def init_camera(self):
        """
        Initiate video capture using the first webcam found.
        Set camera width and height settings.
        """
        self.camera = cv2.VideoCapture(0)
        self.camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH)
        self.camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)

    def get_face_samples(self):
        """
        Get a new frame from camera.
        Process this frame.
        Write to an image file, which is awful, but the only way it worked.
        Display the file on the Frame element of the main window.
        """
        retval, frame = self.camera.read()
        frame = imgutils.crop(frame, 70, 320 - 70, 0, 240)

        frame, found, faces = detect.single_cascade(frame, cascade=detect.CASCADE_FACE, return_faces=True)

        if found:
            self.face_samples_recorded += 1
            x, y, w, h = faces[0]
            self.face_samples.append(frame[y:h, x:w])

        cv2.imwrite(".frame.jpg", frame)
        pixbuf_frame = GdkPixbuf.Pixbuf.new_from_file(".frame.jpg")
        self.FaceFrame.set_from_pixbuf(pixbuf_frame)

        if self.face_samples_recorded < 50:
            return True
        else:
            self.camera.release()
            self.save_face_samples()
            return False

    def save_face_samples(self):
        """
        Save captured face samples to our face database
        """
        outputpath = 'faces/' + self.name
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        for i, face in enumerate(self.face_samples):
            cv2.imwrite(outputpath + '/' + str(i+1) + '.jpg', face)
        cv2.imwrite(outputpath + '/' + str(0) + '.jpg', imgutils.resize(self.face_samples[0], 50, 50))
        self.clean_after_acquisition()

    def clean_after_acquisition(self):
        """
        Make cleaning after adding a new face
        """
        self.face_samples_recorded = 0
        self.face_samples = list()
        pixbuf_frame = GdkPixbuf.Pixbuf.new_from_file("resources/gui/shadow.png")
        self.FaceFrame.set_from_pixbuf(pixbuf_frame)
        self.NewFaceName.set_text("")
        self.NewFaceName.set_sensitive(True)
        self.NewFaceButton.set_sensitive(True)
        self.update_face_manager_database()

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
