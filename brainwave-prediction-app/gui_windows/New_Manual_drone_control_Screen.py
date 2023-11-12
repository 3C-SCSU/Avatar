import PySimpleGUI as sg
import GUI.py as test
def manual_drone_control_window(items, get_drone_action, window1):

    # Define the layout for the Manual Drone Control Page

    # Column layouts for centering"Done.")
    top_center = [
        [sg.Button('Up', size=(8, 2), image_filename="brainwave-prediction-app/images/up.png")]]
    top_right = [[sg.Text('Flight Log')], [sg.Listbox(
        values=[], size=(30, 6), key='LOG')]]
    bottom_center = [
        [sg.Button('Down', size=(8, 2), image_filename="brainwave-prediction-app/images/down.png")]]

    manual_drone_control_layout = [
        [sg.Button('Home', size=(8, 2), image_filename="brainwave-prediction-app/images/home.png"), sg.Push(),
         sg.Column(top_center, pad=((55, 0), (0, 0))), sg.Push(), sg.Column(top_right), ],
        [sg.Button('BrainControl', size=(8, 2), image_filename="brainwave-prediction-app/images/brain.png"), sg.Push(),
         sg.Column(top_center, pad=((55, 0), (0, 0))), sg.Push(), sg.Column(top_right), ],
        [sg.Push(), sg.Push(), sg.Button('Forward', size=(8, 2),
                                         image_filename="brainwave-prediction-app/images/forward.png"), sg.Push(), sg.Push(),],

        [sg.Button('Turn Left', size=(8, 2), image_filename="brainwave-prediction-app/images/turnLeft.png"),
         sg.Button('Left', size=(8, 2),
                   image_filename="brainwave-prediction-app/images/left.png"),
         sg.Button(
             'Stream', image_filename="brainwave-prediction-app/images/drone.png"),
         sg.Button('Right', size=(8, 2),
                   image_filename="brainwave-prediction-app/images/right.png"),
         sg.Button('Turn Right', size=(8, 2), image_filename="brainwave-prediction-app/images/turnRight.png")],

        [sg.Button('Back', size=(8, 2),
                   image_filename="brainwave-prediction-app/images/back.png")],
        [sg.Button('Connect', size=(8, 2), image_filename="brainwave-prediction-app/images/connect.png"), sg.Push(), sg.Push(), sg.Column(bottom_center, pad=((55, 0), (0, 0))), sg.Push(), sg.Button('Takeoff', size=(8, 2), image_filename="brainwave-prediction-app/images/takeoff.png"), sg.Button('Land', size=(8, 2), image_filename="brainwave-prediction-app/images/land.png")]]

    manual_drone_control_window = sg.Window(
        'Manual Drone Control', manual_drone_control_layout, size=(1600, 1600), element_justification='c')

    first_iteration = True

    while True:
        event, values = manual_drone_control_window.read()

        if (first_iteration):
            items.insert(0, "---------- NEW LOG ----------")
            manual_drone_control_window['LOG'].update(values=items)
            first_iteration = False

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
            # items.insert(0, get_drone_action('down'))
            get_drone_action('down')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Forward':
            # Code for moving the drone forward
            items.insert(0, "Forward button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('forward'))
            get_drone_action('forward')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Back':
            # Code for moving the drone back
            items.insert(0, "Back button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('backward'))
            get_drone_action('backward')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Left':
            # Code for moving the drone left
            items.insert(0, "Left button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('left'))
            get_drone_action('left')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Right':
            # Code for moving the drone right
            items.insert(0, "Right button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('right'))
            get_drone_action('right')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Turn Left':
            # Code for turning the drone left
            items.insert(0, "Turn Left button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('turn_left'))
            get_drone_action('turn_left')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Turn Right':
            # Code for turning the drone right
            items.insert(0, "Turn right button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            # items.insert(0, get_drone_action('turn_right'))
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
            # items.insert(0, get_drone_action('land'))
            get_drone_action('land')
            items.insert(0, 'done')
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Home':
            # Code for Home
            manual_drone_control_window.close()
            window1.un_hide()
            break
        elif event == 'BrainControl':
            # Code for Home
            manual_drone_control_window.close()
            window1.un_hide()
            test.brainwave_prediction_window(window1, get_drone_action, test.use_brainflow)
            break
        elif event == 'Connect':
            # Code for Connect
            items.insert(0, "Connect button pressed")
            manual_drone_control_window['LOG'].update(values=items)
            get_drone_action("connect")
            items.insert(0, "Done.")
            manual_drone_control_window['LOG'].update(values=items)
        elif event == 'Stream':
            get_drone_action('stream')

    # items.insert(0, "---------- END LOG ----------")
    window1.un_hide
    manual_drone_control_window.close()
