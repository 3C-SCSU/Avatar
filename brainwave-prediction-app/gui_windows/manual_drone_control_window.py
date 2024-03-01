import PySimpleGUI as sg


#changes method layout to class design for retention of data
class Drone_Control:

    items = []
    first_iteration = True
    def __init__(self):
        first_iteration = True

    #sets the layout and returns a tab for the tabgroup
    def manual_drone_control_window(self, get_drone_action):

        # Define the layout for the Manual Drone Control Page

        # Column layouts for centering"Done.")
        # NEW CHANGES: Added expand_x and expand_y parameters so elements scale with the window, should scale, but buttons are misaligned. 
        top_center = [
            [sg.Button('Up', size=(8, 2), expand_x=True, expand_y=True, image_filename="./images/up.png")]]
        top_right = [[sg.Text('Flight Log')], [sg.Listbox(
            values=[], size=(40, 5), key='LOG')]]
        bottom_center = [
            [sg.Button('Down', size=(8, 2), expand_x=True, expand_y=True, image_filename="./images/down.png")]]

        manual_drone_control_layout = [
            [sg.Button('Home', size=(8, 2), image_filename="./images/home.png"),
            sg.Column(top_center, expand_x=True, expand_y=True, pad=((0, 0), (0, 0))), sg.Column(top_right)],
            #[sg.Push(), sg.Push(), sg.Button('Forward', size=(8, 2),  expand_x=True, expand_y=True,
                                            #image_filename="./images/forward.png"), sg.Push(), sg.Push(),],

            [sg.Button('Forward', size=(8, 2),  expand_x=True, expand_y=True,
                                            image_filename="./images/forward.png")],

            [sg.Button('Turn Left', size=(8, 2), expand_x=True, expand_y=True,image_filename="./images/turnLeft.png"),
            sg.Button('Left', size=(8, 2), expand_x=True, expand_y=True,
                    image_filename="./images/left.png"),
            sg.Button(
                'Stream', expand_x=True, expand_y=True, image_filename="./images/drone.png"),
            sg.Button('Right', size=(8, 2), expand_x=True, expand_y=True,
                    image_filename="./images/right.png"),
            sg.Button('Turn Right', size=(8, 2), expand_x=True, expand_y=True, image_filename="./images/turnRight.png")],

            [sg.Button('Back', size=(8, 2), expand_x=True, expand_y=True,
                    image_filename="./images/back.png")],
            [sg.Button('Connect1', size=(8, 2), image_filename="./images/connect.png"), sg.Column(bottom_center, expand_x=True, expand_y=True, pad=((0, 0), (0, 0))),sg.Button('Takeoff', size=(8, 2), image_filename="./images/takeoff.png"), sg.Button('Land', size=(8, 2), image_filename="./images/land.png")]]


        tab = sg.Tab('Manual Drone Control', manual_drone_control_layout, key='Manual Drone Control')
        return tab

        #manual_drone_control_window = sg.Window(
        #    'Manual Drone Control', manual_drone_control_layout, size=(1600, 1600), element_justification='c')



    #loop to read button presses
    def buttonLoopDrone(self, get_drone_action, window1, event1, values1):
        first_iteration = self.first_iteration
        event = event1
        values = values1

        if (first_iteration):
            self.items.insert(0, "---------- NEW LOG ----------")
            window1['LOG'].update(values=self.items)
            first_iteration = False
            self.first_iteration = False

        
        if event == 'Up':
            # Code for moving the drone up
            get_drone_action('up')
            self.items.insert(0, "Up button pressed")
            window1['LOG'].update(values=self.items)
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Down':
            # Code for moving the drone down
            self.items.insert(0, "Down button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('down'))
            get_drone_action('down')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Forward':
            # Code for moving the drone forward
            self.items.insert(0, "Forward button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('forward'))
            get_drone_action('forward')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Back':
            # Code for moving the drone back
            self.items.insert(0, "Back button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('backward'))
            get_drone_action('backward')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Left':
            # Code for moving the drone left
            self.items.insert(0, "Left button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('left'))
            get_drone_action('left')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Right':
            # Code for moving the drone right
            self.items.insert(0, "Right button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('right'))
            get_drone_action('right')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Turn Left':
            # Code for turning the drone left
            self.items.insert(0, "Turn Left button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('turn_left'))
            get_drone_action('turn_left')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Turn Right':
            # Code for turning the drone right
            self.items.insert(0, "Turn right button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('turn_right'))
            get_drone_action('turn_right')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Takeoff':
            # Code for taking off the drone
            self.items.insert(0, "Takeoff button pressed")
            window1['LOG'].update(values=self.items)
            get_drone_action('takeoff')
            self.items.insert(0, "Done.")
            window1['LOG'].update(values=self.items)
        elif event == 'Land':
            # Code for landing the drone
            self.items.insert(0, "Land button pressed")
            window1['LOG'].update(values=self.items)
            # self.items.insert(0, get_drone_action('land'))
            get_drone_action('land')
            self.items.insert(0, 'done')
            window1['LOG'].update(values=self.items)
        elif event == 'Home':
            # Code for Home
            #manual_drone_control_window.close()
            #window1.un_hide()
            #break
            self.first_iteration = True
        elif event == 'Connect1':
            # Code for Connect
            self.items.insert(0, "Connect button pressed")
            window1['LOG'].update(values=self.items)
            get_drone_action('connect')
            self.items.insert(0, "Done.")
            window1['LOG'].update(values=self.items)
        elif event == 'Stream':
            get_drone_action('stream')
 

        return

        # self.items.insert(0, "---------- END LOG ----------")
        #window1.un_hide
        #manual_drone_control_window.close()
