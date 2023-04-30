import PySimpleGUI as sg
import time
import random
from client.brainflow1 import bciConnection 

#TODO enable imports
#tello imports
from djitellopy import Tello
tello = Tello()

def get_drone_action(action): ##receives action from a GUI button and executes a corresponding Tello-Drone class move action, then returns "Done"

    if action == 'connect':
        #tello.connect()
        print("tello.connect()")
    elif action == 'backward':
        #tello.move_back(30)
        print('tello.move_back(30)')           
    elif action == 'down':
        #tello.move_down(30)
        print('tello.move_down(30)')
    elif action == 'forward':
        #tello.move_forward(30)
        print('tello.move_forward(30)')
    elif action == 'land':
        #tello.land
        print('tello.land')
    elif action == 'left':
        #tello.move_left(30)
        print('tello.move_left(30)')
    elif action == 'right':
        #tello.move_right(30)
        print('tello.move_right(30)')
    elif action == 'takeoff':
        #tello.takeoff
        print('tello.takeoff')
    elif action == 'up':
        #tello.move_up(30)
        print('tello.move_up(30)')
    elif action == 'turn_left':
        #tello.rotate_counter_clockwise(45)
        print('tello.rotate_counter_clockwise(45)')
    elif action == 'turn_right':
        #tello.rotate_clockwise(45)
        print('tello.rotate_clockwise(45)')
    elif action == 'flip': 
        #tello.flip_back()
        print("tello.flip('b')")
    
    #TODO Remove sleep
    time.sleep(2)
    return("Done")

def drone_holding_pattern(): 
    print("Hold forward - tello.move(forward(5)")
    #tello.move_forward(5)
    time.sleep(2)
    print("Hold backward - tello.move(backward(5)")
    #tello.move_backward(5)    

    in_pattern = False
    return (in_pattern) #let calling Window know if it needs to restart Holding Pattern
def use_brainflow(): 
    #print("I'm in use brainflow")
    #Create BCI object
    bci = bciConnection()
    server_response = bci.bciConnectionController()
    return server_response
    #data = bci.read_from_board()
    #bci.send_data_to_server(data)

def holding_pattern_window():
    holding_pattern_layout = [
                                [sg.Text('Holding Pattern Log')],
                                # [sg.Output(s=(45,10))],  

                                [sg.Button('Start Holding Pattern'), sg.Button('Stop Holding Pattern')], 
                             ]
    holding_pattern_window = sg.Window("Holding Pattern", holding_pattern_layout, size=(500, 500), element_justification='c')
    
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

def brainwave_prediction_window():
    #Layout
    flight_log = [] #array to hold flight log info
    predictions_log = [] #array to hold info for table
    predictions_headings = ['Predictions Count', 'Server Predictions', 'Prediction Label'] # table headings 
    response_headings = ['Count', 'Label']
    top_left = [    
                    [sg.Radio('Manual Control', 'pilot', default=True, size=(-20, 1)), sg.Radio('Autopilot', 'pilot',  size=(12, 1))],
                    [sg.Button('Read my mind...', size=(40,5), image_filename="./images/brain.png" )], 
                    #[sg.Text(key="-COUNT-"), sg.Text(key="-PREDICTION-")], 
                    [sg.Text("The model says ...")], 
                    [sg.Table (values=[], headings=response_headings,  auto_size_columns=False, def_col_width=15, justification='center', num_rows=1, key='-SERVER_TABLE-', row_height=25, tooltip="Server Response Table", hide_vertical_scroll=True,)],

                    [sg.Button('Not what I was thinking...', size=(14,3)), sg.Button('Execute', size=(14,3)), sg.Push()],  
               ]
    bottom_left = [
                    [sg.Text('Flight Log')],[sg.Listbox(values=flight_log, size=(30, 6), key='LOG')],
                  ]
    bottom_right = [
                    [sg.Text('Console Log')],
                    [sg.Output(s=(45,10))]  
                   ]

    brainwave_prediction_layout = [                                    
                                    
                                    [sg.Column(top_left, pad=((150,0),(0,0))), sg.Push(), sg.Table(
                                                                    values=[],
                                                                    headings=predictions_headings, 
                                                                    max_col_width=35, 
                                                                    auto_size_columns=True, 
                                                                    justification='center',
                                                                    num_rows=10,
                                                                    key='-TABLE-',
                                                                    row_height=35,
                                                                    tooltip='Predictions Table'
                                                                    )
                                    ], 
                                    [sg.Column(bottom_left), sg.Push(), sg.Column(bottom_right)],
                                    [sg.Button('Connect', size=(8,2), image_filename="./images/connect.png"), sg.Push(),],                               

                                  ]

    brainwave_prediction_window = sg.Window("Brainwave Prediction", brainwave_prediction_layout, size=(1200, 800), element_justification='c')
    

    count = 0
    predictions_list = ['backward', 'down', 'forward', 'land', 'left', 'right', 'takeoff', 'up']
    #actions = {'backward': print("BACKWARD2"), 'down': print("BACKWARD"), 'forward': print("BACKWARD"), 'land': print("BACKWARD"), 'left': print("BACKWARD"), 'right': print("BACKWARD"), 'takeoff': print("BACKWARD"), 'up': print("BACKWARD")}
    action_index = 0

    while True: 

        event, values = brainwave_prediction_window.read() 
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
        elif event == "Read my mind...":
            prediction_response = use_brainflow()
            count = prediction_response['prediction_count']
            prediction_label = prediction_response['prediction_label']        
            #brainwave_prediction_window["-COUNT-"].update(count)
            #brainwave_prediction_window["-PREDICTION-"].update(prediction_label)

            server_record = [[count, prediction_label]]
            brainwave_prediction_window['-SERVER_TABLE-'].update(values=server_record)
        elif event == "Not what I was thinking...":
            print("you pushed NOT")
        elif event == "Execute":
            flight_log.insert(0, prediction_label)
            brainwave_prediction_window['LOG'].update(values=flight_log)
            get_drone_action(prediction_label)
            #execute = get_drone_action(prediction_label)
            print("done")
            flight_log.insert(0, "done")
            brainwave_prediction_window['LOG'].update(values=flight_log)
            prediction_record = [len(predictions_log)+1, count, prediction_label]
            predictions_log.append(prediction_record)
            brainwave_prediction_window['-TABLE-'].update(values=predictions_log)
        elif event == 'Connect':
            # Code for Connect
            flight_log.insert(0, "Connect button pressed")
            brainwave_prediction_window['LOG'].update(values=flight_log)
            get_drone_action('connect')
            flight_log.insert(0, "Done.")
            brainwave_prediction_window['LOG'].update(values=flight_log)

    window1.un_hide()
    brainwave_prediction_window.close() 

def manual_drone_control(items): 

    #Define the layout for the Manual Drone Control Page

    #Column layouts for centering"Done.")
    top_center = [[sg.Button('Up', size=(8,2), image_filename="./images/up.png")]]
    top_right = [[sg.Text('Flight Log')], [sg.Listbox(values=[], size=(30, 6), key='LOG')]]
    bottom_center = [[sg.Button('Down', size=(8,2), image_filename="./images/down.png")]]

    manual_drone_control_layout = [ 
                [sg.Button('Home', size=(8,2), image_filename="./images/home.png"), sg.Push(), sg.Column(top_center, pad=((55,0),(0,0))), sg.Push(), sg.Column(top_right), ],       
                [sg.Push(),sg.Push(),sg.Button('Forward', size=(8,2), image_filename="./images/forward.png"), sg.Push(), sg.Push(),], 

                [sg.Button('Turn Left', size=(8,2), image_filename="./images/turnLeft.png"),  
                sg.Button('Left', size=(8,2), image_filename="./images/left.png"),
                sg.Button(image_filename="./images/drone.png"),
                sg.Button('Right', size=(8,2), image_filename="./images/right.png"),
                sg.Button('Turn Right', size=(8,2), image_filename="./images/turnRight.png")],

                [sg.Button('Back', size=(8,2), image_filename="./images/back.png")],
                [sg.Button('Connect', size=(8,2), image_filename="./images/connect.png"), sg.Push(),sg.Push(), sg.Column(bottom_center, pad=((55,0),(0,0))), sg.Push(), sg.Button('Takeoff', size=(8,2), image_filename="./images/takeoff.png"), sg.Button('Land', size=(8,2), image_filename="./images/land.png")]]
    
    manual_drone_control_window = sg.Window('Manual Drone Control', manual_drone_control_layout, size=(1200, 800), element_justification='c')

    first_iteration = True
    
    while True:
        event, values = manual_drone_control_window.read()

        if(first_iteration):
            items.insert(0, "---------- NEW LOG ----------")
            manual_drone_control_window['LOG'].update(values=items)
            first_iteration=False

        if event == sg.WIN_CLOSED:
            manual_drone_control_window.close()
            window1.un_hide()
            break
        elif event == 'Up':
            # Code for moving the drone up
            get_drone_action('up')
            items.insert(0, "Up button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Down':
            # Code for moving the drone down                
            items.insert(0, "Down button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('down'))
            get_drone_action('down')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Forward':
            # Code for moving the drone forward
            items.insert(0, "Forward button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('forward'))
            get_drone_action('forward')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Back':
            # Code for moving the drone back                
            items.insert(0, "Back button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('backward'))
            get_drone_action('backward')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Left':
            # Code for moving the drone left
            items.insert(0, "Left button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('left'))
            get_drone_action('left')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Right':
            # Code for moving the drone right
            items.insert(0, "Right button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('right'))
            get_drone_action('right')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Turn Left':
            # Code for turning the drone left
            items.insert(0, "Turn Left button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('turn_left'))
            get_drone_action('turn_left')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Turn Right':
            # Code for turning the drone right
            items.insert(0, "Turn right button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('turn_right'))
            get_drone_action('turn_right')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Takeoff':
            # Code for taking off the drone
            items.insert(0, "Takeoff button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            get_drone_action('takeoff')
            items.insert(0, "Done.")
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Land':
            # Code for landing the drone
            items.insert(0, "Land button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            #items.insert(0, get_drone_action('land'))
            get_drone_action('land')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Home':
            # Code for Home
            manual_drone_control_window.close()
            window1.un_hide()
            break               
        elif event == 'Connect':
            # Code for Connect
            items.insert(0, "Connect button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            tello.connect()
            items.insert(0, "Done.")
            manual_drone_control_window['LOG'].update(values=items)
    

    # items.insert(0, "---------- END LOG ----------")
    window1.un_hide
    manual_drone_control_window.close()




# Define the layout for the Starting Page
layout1 = [
            [sg.Button('Brainwave Reading', size=(20,3)), 
            sg.Button('Transfer Data', size=(20,3)),
            sg.Button('Manual Drone Control', size=(20,3)),   
            sg.Button('Holding Pattern', size=(20,3))]          
          ]

# Define the layout for the Brainwave Reading Page
#layout3 = [[sg.Button(image_filename="./images/brain.png")]]

# Define the layout for the Transfer Data Page
layout4 = [[sg.Button(image_filename="./images/upload.png")]]

# Create the windows
window1 = sg.Window('Start Page', layout1, size=(1200, 800))
#window2 = sg.Window('Manual Drone Control', layout2, size=(1200, 800), element_justification='c')
#window3 = sg.Window('Brainwave Reading', layout3, size=(1200, 800), element_justification='c')
window4 = sg.Window('Transfer Data', layout4, size=(1200, 800), element_justification='c')



items = []

# Event loop for the first window
while True:
    event1, values1 = window1.read()
    if event1 == sg.WIN_CLOSED:
        break
    elif event1 == 'Brainwave Reading':
        window1.hide()
        brainwave_prediction_window()
    elif event1 == 'Transfer Data':
        window1.hide()
        window4.read()
    elif event1 == 'Manual Drone Control':
        window1.hide()
        manual_drone_control(items)
    elif event1 == 'Holding Pattern':
        holding_pattern_window()


