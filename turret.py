#!/usr/bin/python3

# Import packages for arguments parsing
import sys
import textwrap
import argparse

# Parse arguments
if sys.platform == "linux" or sys.platform == "linux2":
    parser = argparse.ArgumentParser(description="People detection turret. It detects people and optionally dispenses product.",
                                    epilog=textwrap.dedent('''
                                    .  Available modes:
                                    .  --------------------------------
                                    .  motion:               Motion detection function based on background subtraction.
                                    .  upperbody-face:       Upperbody and face detection
                                    .  face-recognition:     Face detection and recognition

                                '''), formatter_class=argparse.RawDescriptionHelpFormatter,)

parser.add_argument("-s", help="Turn on the turret's sound modules.", action="store_true")
parser.add_argument("-g", help="Show a graphical user interface.", action="store_true")
parser.add_argument("-d", help="Save images on disk hierarchically by date", action="store_true")
parser.add_argument("-r", help="Rotate frame by specified angle")
parser.add_argument("-m", help="The detection mode")

args = parser.parse_args()

# Import standard packages
import os
import array
import signal
import locale
import datetime
import threading

# Import external packages
import cv2
import numpy
from PIL import Image

# Import project packages
from modules import imgutils
from modules import detect
from modules import soundcat
from modules import save

# Set locale (standardize month names to english)
if sys.platform == "linux" or sys.platform == "linux2":
    locale.setlocale(locale.LC_TIME, "en_US.utf8")
elif sys.platform == "win32":
    locale.setlocale(locale.LC_TIME, "usa_usa")

# Turret global variables
SPEAK = args.s or False
GUI = args.g or False
SAVE_TO_DISK = args.d or not GUI
ROTATION = int(args.r or 0)
MODE = args.m or 'motion'

# Frame width and height
WIDTH  = 640
HEIGHT = 480

# OpenCV camera settings
CV_CAP_PROP_FRAME_WIDTH  = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

# Configure camera
camera = None
def init_camera():
    """
    Initiate video capture using the first webcam found.
    Set camera width and height settings.
    """
    global camera
    camera = cv2.VideoCapture(-1)
    camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # Set camera settings
    # sudo apt-get install v4l-utils
    # v4l2-ctl --list-devices
    # v4l2-ctl -d /dev/video0 --list-ctrls
    # v4l2-ctl --set-ctrl=gain_automatic=0
    # v4l2-ctl --set-ctrl=exposure=1000
    # v4l2-ctl --get-ctrl=gain_automatic
    # v4l2-ctl --get-ctrl=exposure
    os.system('v4l2-ctl --set-ctrl=gain_automatic=0')
    os.system('v4l2-ctl --set-ctrl=exposure=1000')

# Configure speaker
speaker = None
def init_speaker():
    """
    Initialize speaker modules
    """
    global speaker
    if SPEAK:
        speaker = soundcat.Soundcat(pps=1.0/5)
        speaker.add_category('init', 'resources/sounds/init')
        speaker.add_category('detected', 'resources/sounds/detected')
        speaker.add_category('quit', 'resources/sounds/quit')

# Convert daily detections to a video
timer = None
def convert_to_video():
    global timer
    now = datetime.datetime.now()
    if timer is None:
        next_convert_time = now
        next_convert_time.replace(hour=23, minute=50, second=0, microsecond=0)
    else:
        next_convert_time = now + datetime.timedelta(hours=12)
        next_convert_time.replace(hour=23, minute=50, second=0, microsecond=0)
        save.video(now)
    timer = threading.Timer((next_convert_time-now).seconds, convert_to_video)
    timer.setDaemon(True)
    timer.start()

# Main operation
def loop():
    """
    Get a new frame from camera.
    Process this frame according to current detection mode.
    """
    global camera

    # Capture frame
    _, frame = camera.read()

    # Rotate if required
    if ROTATION != 0:
        frame = imgutils.rotate_bound(frame, ROTATION)

    found = None

    # Process according to current detection mode
    if MODE is None or MODE == 'motion':
        frame, found = detect.motion_detection(frame, thresh=50, drawboxes=False)
    elif MODE == 'upperbody-face':
        frame, found = detect.double_cascade(frame)
    elif MODE == 'face-recognition':
        frame, found = detect.face_recognition(frame)

    # Save detections
    now = datetime.datetime.now()
    timestr = '%02d/%02d/%04d %02d:%02d:%02d' % (now.day, now.month, now.year, now.hour, now.minute, now.second)
    font = cv2.FONT_HERSHEY_PLAIN
    color= (255, 255, 255) if numpy.mean(frame[0:30,0:120])/255 < 0.6 else (0, 0, 0)
    cv2.putText(frame, timestr, (5, 20), font, 1.2, color, 0, 4)

    if found:
        if SAVE_TO_DISK: save.save(frame, now)
        if SPEAK: speaker.play("detected", use_pps=True)

    return frame


class Cli:
    """
    A class to control CLI operations.
    """

    def __init__(self):
        """
        Initialize camera
        """
        init_camera()
        init_speaker()
        if SPEAK: speaker.play("init")

    def start(self):
        """
        Run loop
        """
        print("Turret is on!")
        while True:
            loop()

class Gui:
    """
    A class to control GUI operations.
    """

    def __init__(self):
        """
        Gui constructor

        Load GUI template from gui.glade file into a Gtk.Builder object.
        Instantiate objects for every dynamic GUI element.
        Initiate video capture.
        Start frame update.
        """
        self.gladefile = 'resources/gui/gui.glade'
        self.gtk = Gtk.Builder()
        self.gtk.add_from_file(self.gladefile)
        self.gtk.connect_signals(self)

        self.Frame = self.gtk.get_object("Frame")
        self.SpeakSwitch = self.gtk.get_object("SpeakSwitch")
        self.SaveToDiskSwitch = self.gtk.get_object("SaveToDiskSwitch")
        self.DetectionModeCombo = self.gtk.get_object("DetectionModeCombo")

        self.init_speak_switch()
        self.init_savetodisk_switch()
        self.init_detectionmode_combo()

        init_camera()
        init_speaker()

        GLib.idle_add(self.update_frame)

        if GUI:
            self.MainWindow = self.gtk.get_object("MainWindow")
            self.MainWindow.connect("delete-event", self.close_button_pressed)
            self.MainWindow.show_all()

        if SPEAK: speaker.play("init")

    def init_speak_switch(self):
        """
        Connect the method update_speak_switch() to SpeakSwitch.
        Set initial state as defined in the command line arguments.
        """
        self.SpeakSwitch.connect("notify::active", self.update_speak_switch)
        self.SpeakSwitch.set_active(SPEAK)

    def update_speak_switch(self, switch, params):
        """
        SPEAK defines if the turret speaker modules are on or off.
        The state of the speak switch updates the global variable SPEAK.
        """
        global SPEAK
        SPEAK = self.SpeakSwitch.get_active()
        init_speaker()

    def init_savetodisk_switch(self):
        """
        Connect the method update_savetodisk_switch() to SaveToDiskSwitch.
        Set initial state as defined in the command line arguments.
        """
        self.SaveToDiskSwitch.connect("notify::active", self.update_savetodisk_switch)
        self.SaveToDiskSwitch.set_active(SAVE_TO_DISK)

    def update_savetodisk_switch(self, switch, params):
        """
        SAVE_TO_DISK defines if the turret saves detections on disk.
        The state of the save on disk switch updates the global variable SAVE_TO_DISK.
        It Also changes the state of cloud backup switches and variables.
        """
        global SAVE_TO_DISK
        SAVE_TO_DISK = self.SaveToDiskSwitch.get_active()

    def init_detectionmode_combo(self):
        """
        Connect the method update_detectionmode_combo() to DetectionModeCombo.
        Set initial state as defined in the command line arguments.
        """
        self.DetectionModeCombo.connect("changed", self.update_detectionmode_combo)
        for mode in detect.detection_modes:
            self.DetectionModeCombo.append(mode, detect.mode_description[mode])
        self.DetectionModeCombo.set_active_id(MODE)

    def update_detectionmode_combo(self, combo):
        """
        MODE defines the detection algorithm our turret is running
        The selected option updates the global variable MODE.
        """
        global MODE
        MODE = self.DetectionModeCombo.get_active_id()

    def close_button_pressed(self, widget, event):
        """
        If close button is pressed, better clean our resources.
        Turn off the camera.
        Execute clean() method, for other resources deallocations.
        Then, actually shut down the poor turret with Gtk.main_quit().
        """
        clean()
        Gtk.main_quit()

    def update_frame(self):
        """
        Calls loop, updates frame
        """
        frame = loop()

        # Convert OpenCV image format to GDK Pixbuf
        h, w, _ = frame.shape
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(frame_show.flatten(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*3, None, None)
        self.Frame.set_from_pixbuf(pixbuf)

        return True

def clean():
    """
    Use this to close the turret's modules when shutting down.
    """
    if SPEAK: speaker.play('quit')
    global camera
    camera.release()

def sigint_handler(signum, instant):
    """
    Capture SIGINT signal and quit safely.
    """
    clean()
    sys.exit()

if __name__ == "__main__":

    # Activate capture of SIGINT (Ctrl-C)
    signal.signal(signal.SIGINT, sigint_handler)

    # Convert detections to video every day
    # convert_to_video()

    # Execute GUI or CLI
    if GUI:

        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk, GdkPixbuf, GObject, GLib, Gio

        try:
            # GObject.threads_init()
            g = Gui()
            Gtk.main()
        except KeyboardInterrupt:
            pass

    else:

        cli = Cli()
        cli.start()
