#!/usr/bin/bash
chmod +x ~/main/run.sh
chmod +x ~/main/bash.sh
chmod +x ~/main/python.py
(crontab -l ; echo "59 23 * * 1 ~/main/run.sh") | crontab -
echo "The files will be automatically shuffled every week"