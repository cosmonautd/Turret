# **Turret**
### Access control using computer vision

This project was started with the purpose of monitoring movement of personnel in the research laboratory I worked at. I did most of the software design and implementation during my supervised internship at the Federal University of Ceará. Turret processes video input from a camera, identifies movement frames, performs face detection and records the moments in which people are passing through the entrances/exits. Besides, I implemented a chatbot module that answers to questions asked via Telegram by using natural language processing. For example, there is support for questions and commands like "Who was in the lab today?" or "Hi, Turret, send me the report for today".

<img src="https://davidborges.xyz/assets/turret1.jpg" alt="Who's there?" width="300"/> <img src="https://davidborges.xyz/assets/turret2.jpg" alt="Who's there again?" width="300"/>

<img src="https://davidborges.xyz/assets/turret3.jpg" alt="Is someone there?" width="300"/> <img src="https://davidborges.xyz/assets/turret4.jpg" alt="Daily Activity" width="300"/>


## Installation
#### Debian/Ubuntu
Install apt-get dependencies
 
```
sudo apt-get update
sudo apt-get install python3 python3-pip python3-gi cmake libboost-all-dev
```

Upgrade pip and install pip dependencies
 
```
pip3 install --upgrade pip
pip3 install --user pygame Pillow face_recognition
```

Install OpenCV for Python 3

Try https://stackoverflow.com/questions/37188623/ubuntu-how-to-install-opencv-for-python3


#### Arch/Manjaro
Upgrade pip and install pip dependencies
 
```
pip install --upgrade pip
pip install pygame Pillow face_recognition
```

Install OpenCV and PyGobject with pacman

```
sudo pacman -S opencv python-gobject python2-gobject gtk3 hdf5
```

#### Install chatbot module
Go to teleturret directory and install Botkit along with its dependencies

```
git clone https://github.com/cosmonautd/Botkit.git botkit
cd botkit
./install.sh
```

Besides that, you must set up a file called config.json
