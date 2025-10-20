# NAO Robot Quick Start

## Setup (5 minutes)

### 1. Install Python 2.7

**Ubuntu/Debian:**
```bash
sudo apt install python2.7
```

**macOS:**
```bash
brew install python@2
```

**Windows:**
- Download from [python.org](https://www.python.org/downloads/release/python-2718/)
- Install to `C:\Python27`

### 2. Download NAOqi SDK

1. Get it from [SoftBank Robotics](https://www.aldebaran.com/en/support/nao-6/downloads-softwares)
2. Extract to project root:
   - Linux: `tar -xzf pynaoqi-*.tar.gz`
   - Windows: Extract with 7-Zip

Your folder structure should be:
```
Avatar/
├── pynaoqi-python2.7-*/
├── NA06_Manual_Control/
├── run_nao.sh
└── GUI5.py
```

### 3. Install Python 3 Requirements

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

### 1. Configure NAO IP

Edit `NA06_Manual_Control/nao_service.py`:
```python
NAO_IP = "192.168.1.100"  # Your NAO's IP address
```

**Find NAO's IP:** Press chest button → NAO will speak it

### 2. Start NAO Service

**Linux/Mac:**
```bash
./run_nao.sh
```

**Windows:**
```cmd
run_nao.bat
```

Keep this terminal running.

### 3. Launch GUI (new terminal)

```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
python GUI5.py
```

### 4. Connect

1. Go to **Manual NAO Control** tab
2. Click **Connect**
3. Check flight log for status

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python 2.7 not found" | Install Python 2.7 (see step 1) |
| "NAOqi SDK not found" | Extract SDK to project root |
| "Connection timeout" | Make sure `run_nao.sh` is running |
| "Cannot connect to NAO" | Verify NAO IP and network connection |
| Port 5000 in use | Kill process: `lsof -ti:5000 \| xargs kill -9` |

## Architecture

```
GUI (Python 3) ←→ NAO Service (Python 2.7) ←→ NAO Robot
   localhost:5000              NAOqi SDK         192.168.x.x:9559
```

The NAO service bridges the Python 3 GUI with the legacy Python 2.7 NAOqi SDK.

## Adding Commands

**1. In `nao_service.py`:**
```python
elif command == "wave":
    motion = ALProxy("ALMotion", NAO_IP, NAO_PORT)
    # Your code here
    return {"status": "success", "message": "NAO waved!"}
```

**2. In `GUI5.py`:**
```python
def waveNao(self):
    result = send_command("wave")
    self.log_message(result['message'])
```

**3. In your QML:**
```qml
Button {
    text: "Wave"
    onClicked: backend.waveNao()
}
```

## Support

- **NAOqi Docs:** http://doc.aldebaran.com/2-8/
- **Issues:** GitHub Issues tab