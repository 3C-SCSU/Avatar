# NAO6 Robot Dances and Moves 🎶🤖

### **Important: Please use Linux based OS or wsl for windows.**

This directory contains scripts to control the NAO6 humanoid robot using Choreograph, Python, and the NAOqi API.

## Setup Instructions

### 1. Install Python Dependencies

You'll need the following Python packages:

```bash
pip install python-dotenv
pip install pytest

# since we are using python 2.7
pip install mock
pip install virtualenv
```
**Note**: 
- You also need the official NAOqi SDK installed for Python.
- NAOqi SDK only works with Python 2.7 for now, if you use Python 3.x, it will not work. NAOqi officially doesn't support Py3.
- You might need to use a Python 2.7 virtual environment for full compatibility.

Run this command in bash (make sure to adjust the SDK path, check `setup_naoqi_env.sh` for more details)

```
source setup_naoqi_env.sh 
```

### 2. Configure the Robot Connection
Create a .env file inside the corresponding project folder (e.g., /gangnam/) with your robot's IP address and port:

```bash
ROBOT_IP=192.168.1.10
ROBOT_PORT=9559
```
 - ROBOT_IP: Your NAO6 robot's IP on your network
 - ROBOT_PORT: Usually 9559 (default NAOqi port)


### 3. Upload Required Media
Some moves may require specific audio files or assets. For example, to perform the Gangnam Style dance:

Upload the `gangnam_style.mp3` file to the robot under:
```bash
/home/nao/music/
```
You can use `scp` or Choreograph to transfer files.


## Project Structure

```
/Avatar/NAO6/
    ├── gangnam/             # Gangnam Style Dance
    │   ├── gangnam_dance.py
    │   └── README.md
    ├── new_move/            # Future moves
    │   ├── new_move.py
    │   └── README.md
    ├── tests/               # pytests for NA06 moves
    │   ├── conftest.py      # testing configs
    │   └── test_robot_controls.py
    ├── robot_controller.py  # Robot initilization
    └── README.md            # You are here
```