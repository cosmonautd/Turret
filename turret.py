#!/usr/bin/python3

import sys
import textwrap
import argparse

# Arguments parsing
if sys.platform == "linux" or sys.platform == "linux2":
    parser = argparse.ArgumentParser(description="People detection turret. Detects people and optionally dispenses product.",
                                    epilog=textwrap.dedent('''
                                ...    Available modes:
                                ...    --------------------------------
                                ...    motion:              Motion detection function based on background subtraction.
                                ...    upperbody-face:      Upperbody and face detection
                                ...    face-recognition:    Face detection and recognition

                                '''), formatter_class=argparse.RawDescriptionHelpFormatter,)
elif sys.platform == "win32":
    parser = argparse.ArgumentParser(description="People detection turret. Detects people and optionally dispenses product.",
                                    epilog=textwrap.dedent('''
                                ...    Available modes:
                                ...    --------------------------------
                                ...    motion:              Motion detection function based on background subtraction.
                                ...    upperbody-face:      Upperbody and face detection

                                '''), formatter_class=argparse.RawDescriptionHelpFormatter,)

parser.add_argument("-s", "--speak", help="Turn on the turret's sound modules.", action="store_true")
parser.add_argument("-g", "--gui", help="Show a graphical user interface.", action="store_true")
parser.add_argument("-d", "--save_to_disk", help="Saves images on disk hierarchically by date (enabled by default in CLI operation)", action="store_true")
parser.add_argument("-G", "--backup_gdrive", help="Saves images on Google Drive", action="store_true")
parser.add_argument("-r", "--rotate", help="Rotates frame by specified angle")
parser.add_argument("-m", "--mode", help="The detection mode")

args = parser.parse_args()

import cv2
import sys
import numpy
import array
import signal
import locale
import datetime
from PIL import Image

from modules import imgutils
from modules import detect
from modules import soundcat
from modules import save

# Set locale (standardize month names)
if sys.platform == "linux" or sys.platform == "linux2":
    locale.setlocale(locale.LC_TIME, "en_US.utf8")
elif sys.platform == "win32":
    locale.setlocale(locale.LC_TIME, "usa_usa")

# Turret global variables
SPEAK = args.speak or False
GUI = args.gui or False
SAVE_TO_DISK = args.save_to_disk or not GUI
BACKUP_GOOGLEDRIVE = args.backup_gdrive and SAVE_TO_DISK
ROTATE = int(args.rotate or 0)
MODE = args.mode or 'motion'

# Width and height of the frames our turret will process
WIDTH  = 640
HEIGHT = 480
# Codes for OpenCV camera settings
CV_CAP_PROP_FRAME_WIDTH  = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

# Managing camera
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

# Soundcat object
sound = None
if SPEAK:
    sound = soundcat.Soundcat(pps=1.0/5)
    sound.add_category('init', 'resources/sounds/init')
    sound.add_category('detected', 'resources/sounds/detected')
    sound.add_category('quit', 'resources/sounds/quit')

# Managing Google Drive backup setting
drive = None
upload = None

def init_googledrive():
    """
    Initiate a Drive and UploadQueue objects for Google Drive backup
    """
    global drive, upload
    # Only execute once
    if drive == None and upload == None:
        # Drive object
        drive = save.Drive()

        # UploadQueue object
        upload = save.UploadQueue(drive)
        upload.start()

# Set initial Google Drive Backup configuration
if BACKUP_GOOGLEDRIVE:
    init_googledrive()

def loop():
    """
    Get a new frame from camera.
    Process this frame.
    Write to an image file, which is awful, but the only way it worked.
    Display the file on the Frame element of the main window.
    """
    global camera

    retval, frame = camera.read()

    if ROTATE != 0:
        frame = imgutils.rotate_bound(frame, ROTATE)

    found = None

    if MODE is None or MODE == 'motion':
        frame, found = detect.motion_detection(frame, thresh=50, drawboxes=False)
    elif MODE == 'upperbody-face':
        frame, found = detect.double_cascade(frame)
    elif MODE == 'face-recognition':
        frame, found = detect.face_recognition(frame)

    if found:
        now = datetime.datetime.now()
        if SAVE_TO_DISK: save.save(frame, now)
        if BACKUP_GOOGLEDRIVE: upload.append(now)
        if SPEAK: sound.play("detected", use_pps=True)
    
    return frame


class Cli:
    """
    A class to control the turret's CLI operations.
    """

    def __init__(self):
        """
        Cli constructor
        """
        init_camera()
        if SPEAK: sound.play("init")
    
    def start(self):
        """ Main Turret operation
        """
        while True:
            loop()

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
        self.SpeakSwitch = self.gtk.get_object("SpeakSwitch")
        self.SaveToDiskSwitch = self.gtk.get_object("SaveToDiskSwitch")
        self.BackupGoogleDriveSwitch = self.gtk.get_object("BackupGoogleDriveSwitch")
        self.DetectionModeCombo = self.gtk.get_object("DetectionModeCombo")

        self.init_speak_switch()
        self.init_savetodisk_switch()
        self.init_backup_googledrive_switch()
        self.init_detectionmode_combo()

        init_camera()

        GLib.idle_add(self.update_frame)

        if GUI:
            self.MainWindow = self.gtk.get_object("MainWindow")
            self.MainWindow.connect("delete-event", self.close_button_pressed)
            self.MainWindow.show_all()

        if SPEAK: sound.play("init")

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

        # Update backup elements. They are active and sensitive only if SAVE_TO_DISK is enabled
        self.BackupGoogleDriveSwitch.set_active(BACKUP_GOOGLEDRIVE and SAVE_TO_DISK)
        self.BackupGoogleDriveSwitch.set_sensitive(SAVE_TO_DISK)

    def init_backup_googledrive_switch(self):
        """
        Connect the method update_backup_googledrive_switch() to BackupGoogleDriveSwitch.
        Set initial state as defined in the command line arguments.
        """
        self.BackupGoogleDriveSwitch.connect("notify::active", self.update_backup_googledrive_switch)
        self.BackupGoogleDriveSwitch.set_active(BACKUP_GOOGLEDRIVE and SAVE_TO_DISK)
        self.BackupGoogleDriveSwitch.set_sensitive(SAVE_TO_DISK)

    def update_backup_googledrive_switch(self, switch, params):
        """
        BACKUP_GOOGLEDRIVE defines if the turret saves detections on Google Drive.
        The state of the Google Drive switch updates the global variable BACKUP_GOOGLEDRIVE.
        """
        global BACKUP_GOOGLEDRIVE, SAVE_TO_DISK
        BACKUP_GOOGLEDRIVE = self.BackupGoogleDriveSwitch.get_active() and SAVE_TO_DISK
        if BACKUP_GOOGLEDRIVE:
            init_googledrive()

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

        h, w, d = frame.shape
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(frame_show.flatten(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w*3, None, None)
        self.Frame.set_from_pixbuf(pixbuf)

        return True

def clean():
    """
    Use this to close the turret's modules when shutting down.
    """
    if SPEAK: sound.play('quit')
    if upload: upload.quit()
    global camera
    camera.release()
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

    if GUI:

        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk, GdkPixbuf, GObject, GLib, Gio

        try:
            GObject.threads_init()
            g = Gui()
            Gtk.main()
        except KeyboardInterrupt:
            pass
    
    else:

        cli = Cli()
        cli.start()
