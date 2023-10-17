import PySimpleGUI as sg

class Brainwaves:


    # Layout
    flight_log = []  # array to hold flight log info
    predictions_log = []  # array to hold info for table
    predictions_headings = [
        'Predictions Count', 'Server Predictions', 'Prediction Label']  # table headings
    response_headings = ['Count', 'Label']
    count = 0
    predictions_list = ['backward', 'down', 'forward',
                        'land', 'left', 'right', 'takeoff', 'up']
     # actions = {'backward': print("BACKWARD2"), 'down': print("BACKWARD"), 'forward': print("BACKWARD"), 'land': print("BACKWARD"), 'left': print("BACKWARD"), 'right': print("BACKWARD"), 'takeoff': print("BACKWARD"), 'up': print("BACKWARD")}
    action_index = 0

    #deleted window1
    def brainwave_prediction_window(self, get_drone_action, use_brainflow):

        flight_log = self.flight_log
        predictions_log = self.predictions_log
        predictions_headings = self.predictions_headings
        response_headings = self.response_headings
        count = self.count
        predictions_list = self.predictions_list
        action_index = self.action_index

        top_left = [
            [sg.Radio('Manual Control', 'pilot', default=True, size=(-20, 1)),
            sg.Radio('Autopilot', 'pilot',  size=(12, 1))],
            [sg.Button('Read my mind...', size=(40, 5),
                    image_filename="brainwave-prediction-app/images/brain.png")],
            # [sg.Text(key="-COUNT-"), sg.Text(key="-PREDICTION-")],
            [sg.Text("The model says ...")],
            [sg.Table(values=[], headings=response_headings,  auto_size_columns=False, def_col_width=15, justification='center',
                    num_rows=1, key='-SERVER_TABLE-', row_height=25, tooltip="Server Response Table", hide_vertical_scroll=True,)],

            [sg.Button('Not what I was thinking...', size=(14, 3)),
            sg.Button('Execute', size=(14, 3)), sg.Push()],
            [sg.Input(key='-drone_input-'),
            sg.Button('Keep Drone Alive')]
        ]
        bottom_left = [
            [sg.Text('Flight Log')], [sg.Listbox(
                values=flight_log, size=(30, 6), key='LOG')],
        ]
        bottom_right = [
            [sg.Text('Console Log')],
            [sg.Output(s=(45, 10))]
        ]

        brainwave_prediction_layout = [

            [sg.Column(top_left, pad=((150, 0), (0, 0))), sg.Push(), sg.Table(
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
            [sg.Column(bottom_left), sg.Push(),
            sg.Column(bottom_right)],
            [sg.Button('Connect', size=(8, 2),
                    image_filename="brainwave-prediction-app/images/connect.png"), sg.Push(),],

        ]

        tab = sg.Tab('Brainwave Reading', brainwave_prediction_layout, key='Brainwave Reading')
        return tab

        #brainwave_prediction_window = sg.Window("Brainwave Prediction", brainwave_prediction_layout, size=(
        #    1200, 800), element_justification='c', finalize=True)


    def buttonLoop(self, window1, event1, values1, get_drone_action, use_brainflow):
        
        flight_log = self.flight_log
        predictions_log = self.predictions_log
        predictions_headings = self.predictions_headings
        response_headings = self.response_headings
        count = self.count
        predictions_list = self.predictions_list
        action_index = self.action_index
            
        event = event1
        values = values1
           

        if event == "Read my mind...":
            prediction_response = use_brainflow()
            count = prediction_response['prediction_count']
            prediction_label = prediction_response['prediction_label']
            # brainwave_prediction_window["-COUNT-"].update(count)
            # brainwave_prediction_window["-PREDICTION-"].update(prediction_label)

            server_record = [[count, prediction_label]]
            window1['-SERVER_TABLE-'].update(
                values=server_record)
        elif event == "Not what I was thinking...":
            get_drone_action(values['-drone_input-'])
            prediction_record = [
                "manual", "predict", f"{values['-drone_input-']}"]
            predictions_log.append(prediction_record)
            window1['-TABLE-'].update(
                values=predictions_log)
        elif event == "Execute":
            flight_log.insert(0, prediction_label)
            window1['LOG'].update(values=flight_log)
            get_drone_action(prediction_label)
            # execute = get_drone_action(prediction_label)
            print("done")
            flight_log.insert(0, "done")
            window1['LOG'].update(values=flight_log)
            prediction_record = [
                len(predictions_log)+1, count, prediction_label]
            predictions_log.append(prediction_record)
            window1['-TABLE-'].update(
               values=predictions_log)
        elif event == 'Connect':
            # Code for Connect
            flight_log.insert(0, "Connect button pressed")
            window1['LOG'].update(values=flight_log)
            get_drone_action('connect')
            flight_log.insert(0, "Done.")
            window1['LOG'].update(values=flight_log)
        elif event == 'Keep Drone Alive':
            get_drone_action('keep alive')

        return

        #window1.un_hide()
        #brainwave_prediction_window.close()
