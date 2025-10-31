import PySimpleGUI as sg
import time
import random
import cv2
import linecache
import sys
from client.brainflow1 import bciConnection

from gui_windows.manual_drone_control_window import Drone_Control
from gui_windows.brainwave_prediction_window import Brainwaves
from gui_windows.transfer_files_window import TransferData

# TODO enable imports
# tello imports
from djitellopy import Tello

tello = Tello()
# tello.takeoff()


# receives action from a GUI button and executes a corresponding Tello-Drone class move action, then returns "Done"


def get_drone_action(action):

    if action == 'connect':
        tello.connect()
        print("tello.connect()")
    elif action == 'backward':
        tello.move_back(30)
        print('tello.move_back(30)')
    elif action == 'down':
        tello.move_down(30)
        print('tello.move_down(30)')
    elif action == 'forward':
        tello.move_forward(30)
        print('tello.move_forward(30)')
    elif action == 'land':
        tello.land()
        print('tello.land')
    elif action == 'left':
        tello.move_left(30)
        print('tello.move_left(30)')
    elif action == 'right':
        tello.move_right(30)
        print('tello.move_right(30)')
    elif action == 'takeoff':
        tello.takeoff()
        print('tello.takeoff')
    elif action == 'up':
        tello.move_up(30)
        print('tello.move_up(30)')
    elif action == 'turn_left':
        tello.rotate_counter_clockwise(45)
        print('tello.rotate_counter_clockwise(45)')
    elif action == 'turn_right':
        tello.rotate_clockwise(45)
        print('tello.rotate_clockwise(45)')
    elif action == 'flip':
        tello.flip_back()
        print("tello.flip('b')")
    elif action == 'keep alive':
        bat = tello.query_battery()
        print(bat)
    elif action == 'stream':
        tello.streamon()
        frame_read = tello.get_frame_read()
        while True:
            print("truu")
            img = frame_read.frame
            cv2.imshow("drone", img)
    else:
        print ("No action supplied!")

    # TODO Remove sleep
    # time.sleep(2)
    return ("Done")


def drone_holding_pattern():
    print("Hold forward - tello.move(forward(5)")
    tello.move_forward(5)
    time.sleep(2)
    print("Hold backward - tello.move(backward(5)")
    tello.move_backward(5)

    in_pattern = False
    # let calling Window know if it needs to restart Holding Pattern
    return (in_pattern)


def use_brainflow():
    # Create BCI object
    bci = bciConnection()
    server_response = bci.bciConnectionController()
    return server_response


def holding_pattern_window():
    holding_pattern_layout = [
        [sg.Text('Holding Pattern Log')],
        # [sg.Output(s=(45,10))],

        [sg.Button('Start Holding Pattern'),
         sg.Button('Stop Holding Pattern')],
    ]
    holding_pattern_window = sg.Window(
        "Holding Pattern", holding_pattern_layout, size=(500, 500), element_justification='c')

    should_hold = False
    in_pattern = False
    resume_hold = False

    while True:
        event, values = holding_pattern_window.read()
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
        elif event == "Start Holding Pattern":
            should_hold = True
            if should_hold and not in_pattern:
                in_pattern = drone_holding_pattern()
            print("Should hold")
        elif event == "Stop Holding Pattern":
            if resume_hold:
                resume_hold = False
                holding_pattern_window['Start Holding Pattern'].click()
            else:
                should_hold = False
                print("!should hold")

        if should_hold and not in_pattern:
            holding_pattern_window['Stop Holding Pattern'].click()
            resume_hold = True
    holding_pattern_window.close()

#COMMENTS FOR ISSUE 39: THESE ARE THE BUTTONS THAT NEED TO CHANGE (start here) ---------
#LAYOUT1 = PySimpleGui is called as 'sg'
#Syntax for the button is sg.Button('text of button', size = (characters wide, characters tall)
#.read() = save values as a tuple (event, values)
#An event is pressing a button or closing the window. 
#We'll need a tab group, tabs for the individual items (so 4)
#And a way of linking the content to each tab.

#added tabs for the tabgroup
brainwaveObj = Brainwaves(get_drone_action)
brainwave_tab = brainwaveObj.brainwave_prediction_window(get_drone_action, use_brainflow)

transferDataObj = TransferData ()
transferData_tab = transferDataObj.transfer_files_window ()

DroneControlObj = Drone_Control()
manDroneCtrlTab = DroneControlObj.manual_drone_control_window(get_drone_action)

t4Test = sg.Text('Disabled for now',text_color='Red')
holdPatTab = [[t4Test]]

#new layout designed
layout1 = [[sg.TabGroup([[
    brainwave_tab,
    transferData_tab,
    manDroneCtrlTab,
    sg.Tab('Holding Pattern', holdPatTab,key='Holding Pattern')]],
    key='layout1',enable_events=True)]]

# Create the windows
window1 = sg.Window('Start Page', layout1, size=(1200,800),element_justification='c',resizable=True,finalize=True)
# window1.Maximize()

# Event loop for the first window
while True:
    event1, values1 = window1.read()
    activeTab = window1['layout1'].Get()
    
    try:
        if event1 == sg.WIN_CLOSED:
            break
        elif activeTab == 'Brainwave Reading':
            brainwaveObj.buttonLoop(window1, event1, values1, get_drone_action, use_brainflow)
        elif activeTab == 'Transfer Data':
            transferDataObj.buttonLoop (window1, event1, values1)
        elif activeTab == 'Manual Drone Control':
            #window1.hide()
            DroneControlObj.buttonLoopDrone(get_drone_action, window1, event1, values1)
        #elif activeTab == 'Holding Pattern':
            #holding_pattern_window()
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache (filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        errorstr = 'EXCEPTION IN ({},\nLINE {}\n"{}"):\n\n{} {}'.format(filename, lineno, line.strip(), type(e), exc_obj)

        print (type(e), exc_obj)
        sg.popup_error(errorstr)