import cv2
import sys
import signal
from PIL import Image
from gi.repository import Gtk, GdkPixbuf, GObject, GLib, Gio

from modules import detect
from modules import imgutils

# Width and height of the frames Face Manager will process
WIDTH  = 320;
HEIGHT = 240;
# Codes for OpenCV camera settings
CV_CAP_PROP_FRAME_WIDTH  = 3;
CV_CAP_PROP_FRAME_HEIGHT = 4;

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

        self.init_newface_name()
        self.init_newface_button()

        self.MainWindow = self.gtk.get_object("MainWindow")
        self.MainWindow.connect("delete-event", self.close_button_pressed)
        self.MainWindow.show_all()

        self.face_samples_recorded = 0

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
        self.camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
        self.camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

    def get_face_samples(self):
        """
        Get a new frame from camera.
        Process this frame.
        Write to an image file, which is awful, but the only way it worked.
        Display the file on the Frame element of the main window.
        """
        retval, frame = self.camera.read()
        frame = imgutils.crop(frame, 70, 320 - 70, 0, 240)

        frame, found = detect.single_cascade(frame, detect.CASCADE_FACE)

        if found:
            self.face_samples_recorded += 1

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
        """
        pass
        self.clean_after_acquisition()

    def clean_after_acquisition(self):
        self.face_samples_recorded = 0
        pixbuf_frame = GdkPixbuf.Pixbuf.new_from_file("resources/gui/shadow.png")
        self.FaceFrame.set_from_pixbuf(pixbuf_frame)
        self.NewFaceName.set_text("")
        self.NewFaceName.set_sensitive(True)
        self.NewFaceButton.set_sensitive(True)

    def close_button_pressed(self, widget, event):
        """
        If close button is pressed, better clean our resources.
        Turn off the camera.
        Execute clean() method, for other resources deallocations.
        Then, actually shut down the poor turret with Gtk.main_quit().
        """
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
