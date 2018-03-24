
## **Turret** ##

#### Install on Debian/Ubuntu ####

 - Install apt-get dependencies
 
```
sudo apt-get update
sudo apt-get install python3 python3-pip python3-gi cmake libboost-all-dev
```

 - Upgrade pip and install pip dependencies
 
```
pip3 install --upgrade pip
pip3 install --user pygame Pillow face_recognition
```

 - Install OpenCV for Python 3
 
 Try https://stackoverflow.com/questions/37188623/ubuntu-how-to-install-opencv-for-python3


#### Install on Arch ####

 - Upgrade pip and install pip dependencies
 
```
pip install --upgrade pip
pip install pygame Pillow face_recognition
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
