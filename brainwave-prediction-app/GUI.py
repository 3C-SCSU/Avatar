import PySimpleGUI as sg
import time
import random
import cv2
from client.brainflow1 import bciConnection

from gui_windows.manual_drone_control_window import Drone_Control
from gui_windows.brainwave_prediction_window import Brainwaves


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
        tello.land
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


# Define the layout for the Starting Page

#COMMENTS FOR ISSUE 39: THESE ARE THE BUTTONS THAT NEED TO CHANGE (start here) ---------
#LAYOUT1 = PySimpleGui is called as 'sg'
#Syntax for the button is sg.Button('text of button', size = (characters wide, characters tall)
#.read() = save values as a tuple (event, values)
#An event is pressing a button or closing the window. 
#We'll need a tab group, tabs for the individual items (so 4)
#And a way of linking the content to each tab.

brainwaveObj = Brainwaves(get_drone_action)
brainwave_tab = brainwaveObj.brainwave_prediction_window(get_drone_action, use_brainflow)

t2Test = sg.Text('Disabled for now',text_color='Red')
transferTab = [[t2Test]]

items = []
DroneControlObj = Drone_Control()
manDroneCtrlTab = DroneControlObj.manual_drone_control_window(items, get_drone_action)

t4Test = sg.Text('Disabled for now',text_color='Red')
holdPatTab = [[t4Test]]


layout1 = [[sg.TabGroup([[
    brainwave_tab,
    sg.Tab('Transfer Data', transferTab, key='Transfer Data'),
    manDroneCtrlTab,
    sg.Tab('Holding Pattern', holdPatTab,key='Holding Pattern')]],
    key='layout1',enable_events=True)]]

OLDlayout1 = [
    [sg.Button('Brainwave Reading', size=(20, 3)),
     sg.Button('Transfer Data', size=(20, 3), disabled=True),
     sg.Button('Manual Drone Control', size=(20, 3)),
     sg.Button('Holding Pattern', size=(20, 3), disabled=True)]
]

# Define the layout for the Transfer Data Page
layout4 = [[sg.Button(
    image_filename="/Users/williamdoyle/Documents/GitHub/Avatar/brainwave-prediction-app/images")]]

# Create the windows
window1 = sg.Window('Start Page', layout1, size=(1200, 800),element_justification='c',finalize=True)
window4 = sg.Window('Transfer Data', layout4, size=(
    1200, 800), element_justification='c')


# Event loop for the first window
while True:
    event1, values1 = window1.read()
    activeTab = window1['layout1'].Get()
    if event1 == sg.WIN_CLOSED:
        break
    elif activeTab == 'Brainwave Reading':
        brainwaveObj.buttonLoop(window1, event1, values1, get_drone_action, use_brainflow)
    #elif activeTab == 'Transfer Data':
        #window1.hide()
        #window4.read()
    elif activeTab == 'Manual Drone Control':
        #window1.hide()
        DroneControlObj.buttonLoopDrone(items, get_drone_action, window1)
    #elif activeTab == 'Holding Pattern':
        #holding_pattern_window()

