#!/bin/bash
# Auto-generated environment fix script

# Install Python 2.7
echo 'Installing Python 2.7...'
curl https://pyenv.run | bash
pyenv install 2.7.18

# Recreate python2_env
echo 'Recreating python2_env...'
virtualenv -p python2.7 /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/python2_env

# Recreate python3_env
echo 'Recreating python3_env...'
python3 -m venv /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/python3_env

# Recreate avatar_env
echo 'Recreating avatar_env...'
python3 -m venv /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/avatar_env

# Make run_avatar.sh executable
chmod +x /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/run_avatar.sh

# Make run_nao6.sh executable
chmod +x /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/run_nao6.sh

# Make run_drone.sh executable
chmod +x /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/run_drone.sh

# Make run_all.sh executable
chmod +x /home/kaliii/Documents/Avatar-Prashant-forked/Avatar/python_2and3/run_all.sh

echo 'Environment fixes completed!'
