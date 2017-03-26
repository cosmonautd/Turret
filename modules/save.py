"""
Upload images to the cloud in a hierarchical time structure.
"""
# coding: utf-8

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os.path
import sys, time, mimetypes
import socket
import cv2
from collections import deque
import threading
import time

def save(img, img_time, uploadqueue=None):
    """Save images to disc or a Google Drive account.

        Save an image in a hierarchical structure inside the detected/
        folder -> year/month/day/image Additionally saves the image
        using a similar structure inside a Google Drive account, if set.

        Args:
            img: a cv2 image.
            img_time: the time of capture.
            google: a Drive object.

        Returns:

        Raises:

    """


    if os.path.exists("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day)))):
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:19] + "s" + ".png")), img);
    elif os.path.exists("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B')))):
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:19] + "s" + ".png")), img);
    elif os.path.exists("/".join(("detected", str(img_time.year)))):
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:19] + "s" + ".png")), img);
    elif os.path.exists("detected"):
        os.mkdir("/".join(("detected", str(img_time.year))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:19] + "s" + ".png")), img);
    else:
        os.mkdir("detected")
        os.mkdir("/".join(("detected", str(img_time.year))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'))))
        os.mkdir("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day))))
        cv2.imwrite("/".join(("detected", str(img_time.year), str(img_time.month) + ". " + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] \
                                                                                                                                                                                            + "m" + str(img_time)[17:19] + "s" + ".png")), img);

    if uploadqueue:
        uploadqueue.append(img_time);

class UploadQueue(object):
    """Implements a queue of images for upload.
    """

    def __init__(self, drive):
        """UploadQueue constructor.

            Args:
                drive: a Drive object.

            Returns:

            Raises:

        """
        self.uploadqueue = deque();
        self.drive = drive;
        self.running = True;

    def append(self, img_time):
        """Append a new image for upload.

            Args:
                img_time: a datetime object representing the time the frame was taken.

            Returns:

            Raises:

        """
        if len(self.uploadqueue) == 0:
            print(str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s", "[ADDED TO UPLOAD LIST]")
            self.uploadqueue.append(img_time)
        elif (not str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s"
                    == str(self.uploadqueue[-1])[:10] + " " + str(self.uploadqueue[-1])[11:13] + "h" + str(self.uploadqueue[-1])[14:16] + "m" + str(self.uploadqueue[-1])[17:19] + "s"):
            print(str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s" , "[ADDED TO UPLOAD LIST]")
            self.uploadqueue.append(img_time)


    def start(self):
        threading.Thread(target=self.uploadloop, args=()).start()

    def uploadloop(self):
        """UploadQueue main method. Continuously uploads the first element of the queue.

            Args:

            Returns:

            Raises:

        """
        print("Starting upload loop")
        while self.running and self.drive:
            if self.drive and len(self.uploadqueue) > 0:
                img_time = self.uploadqueue[0];
                self.uploadqueue.popleft();
                upload_path = None;
                while not upload_path and self.running:
                    upload_path = self.drive.get_link(img_time);
                    time.sleep(0.1);
                if self.drive:
                    disk_path = "/".join(("detected", str(img_time.year), str(img_time.month) + ". "
                                         + img_time.strftime('%B'), str(img_time.day), str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s" + ".png"))
                    print(str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s", "[UPLOAD STARTED]")
                    self.drive.save_img(disk_path, upload_path);
                    print(str(img_time)[:10] + " " + str(img_time)[11:13] + "h" + str(img_time)[14:16] + "m" + str(img_time)[17:19] + "s", "[UPLOAD FINISHED]")
            time.sleep(0.1)

    def quit(self):
        self.drive = None;
        self.running = False;


class Drive(object):
    """Upload images to a Google Drive account in a time structure.

        Drive uses google login data to manage a Google Docs
        account. This same API can be used to manage the new Google
        Drive. This class saves images according to date and time of
        capture in a hierarchical time structure year/month/day/image.

        >>> import save;
        >>> drive = save.Drive();
        >>> current_time = datetime.datetime.now();
        >>> folder_link = drive.get_link(current_time);
        >>> drive.save_img("path/to/image.jpg", folder_link);

        Attributes:
            No public attributes.

    """

    def __init__(self):
        """Drive constructor.

            Args:
                None

            Returns:
                A Drive object.

            Raises:
                No information.

        """
        # TODO: Explain https://googledrive.github.io/PyDrive/docs/build/html/quickstart.html
        # When exception happens here.
        self.g = GoogleAuth()
        self.g.LocalWebserverAuth()
        self.googledrive = GoogleDrive(self.g)


    def folder_exists(self, folder_name, file_list):
        for f in file_list:
            if f['mimeType']=='application/vnd.google-apps.folder' and f['title'] == folder_name:
                return True, f
        return False, None;


    def create_subfolder(self, folder_id, sfldname):
        new_folder = self.googledrive.CreateFile({'title':'{}'.format(sfldname),
                                   'mimeType':'application/vnd.google-apps.folder'})
        if folder_id is not None:
            new_folder['parents'] = [{u'id': folder_id}]
        new_folder.Upload()
        return new_folder


    def get_link(self, img_time, root='root', pos=0):
        """Get the correct link to save an image in the time structure.

            Args:
                time: datetime formatted date object. For example,
                      from datetime.datetime.now().

            Returns:
                A link to the matching folder in the specified Google
                Drive account, according to time. The matching folder
                is Turret/year/month/day.

            Raises:
                No information.

        """

        position = pos
        structure = ['Turret', 'year', 'month', 'day']

        if   structure[position] == 'Turret': folder_name = 'Turret'
        elif structure[position] == 'year':     folder_name = str(img_time.year)
        elif structure[position] == 'month':    folder_name = str(img_time.month) + ". " + img_time.strftime('%B')
        elif structure[position] == 'day':      folder_name = str(img_time.day)

        file_list = None

        try:
            file_list = self.googledrive.ListFile({'q': "'%s' in parents and trashed=false" % root}).GetList()
        except Exception as exception:
            print (exception)

        if file_list:
            exists, folder = self.folder_exists(folder_name, file_list)

            if exists:
                if structure[position] == 'day':
                    return folder
                else:
                    return self.get_link(img_time, root=folder['id'], pos=position+1)
            else:
                new_folder = self.create_subfolder(root, folder_name)
                if structure[position] == 'Turret':
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.year))
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.month) + ". " + img_time.strftime('%B'))
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.day))
                    return new_folder
                elif structure[position] == 'year':
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.month) + ". " + img_time.strftime('%B'))
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.day))
                    return new_folder
                elif structure[position] == 'month':
                    new_folder = self.create_subfolder(new_folder['id'], str(img_time.day))
                    return new_folder
                elif structure[position] == 'day':
                    return new_folder


    def save_img(self, img_path, folder):
        """Save a file to a specified folder in a Google Drive account.

            Args:
                img_path: string, a path to the file that should be
                          uploaded.
                uri: a link to a folder in the desired account.
                     Use get_path().

            Returns:
                Nothing.

            Raises:
                No information.

        """

        new_img = self.googledrive.CreateFile({'title':'{}'.format(os.path.basename(img_path))})
        new_img.SetContentFile(img_path)
        if folder is not None:
            new_img['parents'] = [{u'id': folder['id']}]
        new_img.Upload()
        return new_img
