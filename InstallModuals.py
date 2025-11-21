#program:       InstallModuals
#purpose:       use pip to install moduals requierd to run the aplication
#progamer:      Madison Arndt 1/24/2024


import pip

importlsl = ['opencv-python',
             "paramiko<3.0",
             'pandas',
             'requests',
             'brainflow',
             'pysftp',
             'djitellopy',
             'setuptools',
             'PyQt5',
             "PySide6",
             "pdf2image",
             "serial",
             "pyserial",
             "pysftp"]


for i in importlsl:
    try:
        pip.main(['install', i])
        print()
    except Exception as e:
        print(f"faild to install {i}\n")
















