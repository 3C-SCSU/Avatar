#!/bin/bash

# Universal NAO Service Launcher
# Auto-detects paths - works for all team members

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== NAO Service Launcher ===${NC}\n"

# 1. Detect project root (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "Project root: $PROJECT_ROOT"

# 2. Find NAOqi SDK (auto-detect in project directory)
NAOQI_SDK=""
if [ -d "$PROJECT_ROOT"/pynaoqi-python2.7* ]; then
    NAOQI_SDK=$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name "pynaoqi-python2.7*" | head -n 1)
elif [ -d "$PROJECT_ROOT/naoqi" ]; then
    NAOQI_SDK="$PROJECT_ROOT/naoqi"
elif [ ! -z "$NAOQI_HOME" ]; then
    NAOQI_SDK="$NAOQI_HOME"
else
    echo -e "${RED}ERROR: NAOqi SDK not found!${NC}"
    echo "Please either:"
    echo "  1. Extract NAOqi SDK to project root, OR"
    echo "  2. Set NAOQI_HOME environment variable"
    echo ""
    echo "Download from: https://www.aldebaran.com/en/support/nao-6/downloads-softwares"
    exit 1
fi

echo -e "${GREEN}${NC} Found NAOqi SDK: $NAOQI_SDK"

# 3. Find Python 2.7
PYTHON27=""
if command -v python2.7 &> /dev/null; then
    PYTHON27=$(command -v python2.7)
elif [ -f "/opt/python2.7/bin/python2.7" ]; then
    PYTHON27="/opt/python2.7/bin/python2.7"
elif [ -f "/usr/bin/python2.7" ]; then
    PYTHON27="/usr/bin/python2.7"
elif [ -f "/usr/local/bin/python2.7" ]; then
    PYTHON27="/usr/local/bin/python2.7"
elif command -v python2 &> /dev/null; then
    VERSION=$(python2 --version 2>&1 | grep -oP '2\.7\.\d+' || echo "")
    if [ ! -z "$VERSION" ]; then
        PYTHON27=$(command -v python2)
    fi
else
    echo -e "${RED}ERROR: Python 2.7 not found!${NC}"
    echo "Install instructions:" 
    echo "  Ubuntu/Debian: sudo apt install python2.7"
    echo "  macOS:         brew install python@2"
    exit 1
fi

echo -e "${GREEN}${NC} Found Python 2.7: $PYTHON27"

# 4. Set environment variables
export PYTHONPATH="$NAOQI_SDK/lib/python2.7/site-packages:$PYTHONPATH"
export QI_SDK="$NAOQI_SDK"
export LD_LIBRARY_PATH="$NAOQI_SDK/lib:$LD_LIBRARY_PATH"

# 5. Verify NAO service file exists
SERVICE_FILE="$PROJECT_ROOT/NAO6/nao_service.py"
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}ERROR: nao_service.py not found!${NC}"
    echo "Expected location: $SERVICE_FILE"
    echo "Current directory contents:"
    ls -la "$PROJECT_ROOT"
    exit 1
fi

echo -e "${GREEN}${NC} Found service file: $SERVICE_FILE"

# 6. Check for required NAOqi files
echo ""
echo "Checking NAOqi installation..."
if [ -f "$NAOQI_SDK/lib/python2.7/site-packages/_qi.so" ] || [ -f "$NAOQI_SDK/lib/python2.7/site-packages/qi.so" ]; then
    echo -e "${GREEN}${NC} NAOqi binaries present"
else
    echo -e "${YELLOW}${NC}  Warning: NAOqi binaries not found (will attempt anyway)"
fi

# 7. Kill existing service on port 5000
PORT=5000
echo ""
echo "Checking port $PORT..."
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:$PORT 2>/dev/null || echo "")
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}${NC}  Killing existing service (PID: $PID)"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
elif command -v netstat &> /dev/null; then
    PID=$(netstat -tulpn 2>/dev/null | grep ":$PORT" | awk '{print $7}' | cut -d'/' -f1 || echo "")
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}${NC}  Killing existing service (PID: $PID)"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
fi

# 8. Display configuration
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  NAO Service Port: $PORT"
echo "  Python 2.7:       $PYTHON27"
echo "  NAOqi SDK:        $NAOQI_SDK"
echo "  Service Script:   $SERVICE_FILE"

# 9. Start the service
echo ""
echo -e "${GREEN}Starting NAO service...${NC}"
echo "Press Ctrl+C to stop"
echo ""

$PYTHON27 "$SERVICE_FILE"
