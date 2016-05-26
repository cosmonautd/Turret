import cv2
import sys
import array
import signal
import argparse
import textwrap
from PIL import Image
from gi.repository import Gtk, GdkPixbuf, GObject, GLib, Gio

from modules import detect
from modules import soundcat

# Arguments parsing
parser = argparse.ArgumentParser(description="People detection turret. Detects people and optionally dispenses product.",
                                    epilog=textwrap.dedent('''
                                ...    Available modes:
                                ...    --------------------------------
                                ...    default:             Upperbody and face detection
                                ...    motion-detection:    Motion detection function based on background subtraction.

                                '''), formatter_class=argparse.RawDescriptionHelpFormatter,)
parser.add_argument("-s", "--silent", help="Shut down the turret's sound modules.", action="store_true");
parser.add_argument("-g", "--gui", help="Show a graphical user interface.", action="store_true");
parser.add_argument("-m", "--mode", help="The detection mode")

args = parser.parse_args();

# Turret global variables
SILENT = args.silent
MODE = args.mode

# Width and height of the frames our turret will process
WIDTH  = 640;
HEIGHT = 480;
# Codes for OpenCV camera settings
CV_CAP_PROP_FRAME_WIDTH  = 3;
CV_CAP_PROP_FRAME_HEIGHT = 4;

# Soundcat object
sound = soundcat.Soundcat(pps=1.0/5)
sound.add_category('init', 'resources/sounds/init')
sound.add_category('detected', 'resources/sounds/detected')
sound.add_category('quit', 'resources/sounds/quit')

class Gui:
    """
    A class to control the turret's GUI operations.
    """

    def __init__(self):
        """
        Gui constructor

        Load GUI template from gui.glade file into a Gtk.Builder object.
        Intantiate objects for each necessary GUI element.
        Initiate video capture.
        Start frame update.
        """
        self.gladefile = 'resources/gui/gui.glade'
        self.gtk = Gtk.Builder()
        self.gtk.add_from_file(self.gladefile)
        self.gtk.connect_signals(self)

        self.Frame = self.gtk.get_object("Frame")
        self.SilentSwitch = self.gtk.get_object("SilentSwitch")

        self.init_silent_switch()

        self.init_camera()
        self.last_frame = None

        GLib.idle_add(self.update_frame)

        if args.gui:
            self.MainWindow = self.gtk.get_object("MainWindow")
            self.MainWindow.connect("delete-event", self.close_button_pressed)
            self.MainWindow.show_all()

        if not SILENT: sound.play("init")

    def init_silent_switch(self):
        """
        Connect the method update_silent_switch() to SilentSwitch.
        Set initial state as defined in the command line arguments.
        """
        self.SilentSwitch.connect("notify::active", self.update_silent_switch)
        self.SilentSwitch.set_active(SILENT)

    def update_silent_switch(self, switch, params):
        """
        SILENT defines if the turret speaker modules are on or off.
        The state of the silent switch updates the global variable SILENT.
        """
        global SILENT
        if self.SilentSwitch.get_active(): SILENT = True
        else: SILENT = False

    def init_camera(self):
        """
        Initiate video capture using the first webcam found.
        Set camera width and height settings.
        """
        self.camera = cv2.VideoCapture(0)
        self.camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
        self.camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

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

    def update_frame(self):
        """
        Get a new frame from camera.
        Process this frame.
        Write to an image file, which is awful, but the only way it worked.
        Display the file on the Frame element of the main window.
        """
        retval, frame = self.camera.read()

        if MODE == None or MODE == 'default':
            frame, decision = detect.old_detection(frame);
        elif MODE == 'motion-detection':
            if self.last_frame == None:
                self.last_frame = frame
            frame, self.last_frame, decision = detect.motion_detection(frame, self.last_frame);

        if decision == True:
            if not SILENT: sound.play("detected", use_pps=True)

        cv2.imwrite(".frame.jpg", frame)
        pixbuf_frame = GdkPixbuf.Pixbuf.new_from_file(".frame.jpg")
        self.Frame.set_from_pixbuf(pixbuf_frame)
        return True

def clean():
    """
    Use this to close the turret's modules when shutting down.
    """
    if not SILENT: sound.play('quit')
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
        g = Gui()
        Gtk.main()
    except KeyboardInterrupt:
        pass
