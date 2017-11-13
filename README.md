
## **Turret** ##

#### Install on Arch ####

 - Upgrade pip and install some dependencies
 
```
pip install --upgrade pip
pip install pygame Pillow PyDrive face_recognition
```
 - Install OpenCV and PyGobject with pacman
```
sudo pacman -S opencv python-gobject python2-gobject gtk3 dlib
```
 - Install dlib for Python
```
git clone https://github.com/davisking/dlib.git
cd dlib
sudo python setup.py install
```

#### Install on Windows ####

 - Download and install Python 3.4.x (https://www.python.org/downloads/release/python-344/)
 - Download and install latest PyGObject for Windows (https://sourceforge.net/projects/pygobjectwin32/)
 - Upgrade pip and install some dependencies
```
python -m pip install --upgrade pip
pip install opencv-python pygame Pillow PyDrive
```
