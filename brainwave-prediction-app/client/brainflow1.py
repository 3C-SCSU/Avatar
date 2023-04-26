import time
import numpy as np
import pandas as pd
import requests
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter


class bciConnection():

    def read_from_board(self):

        # print(BoardIds.CYTON_BOARD.value)

        # use synthetic board for demo
        # params = BrainFlowInputParams()
        # params.serial_port = "/dev/cu.usbserial-D200PMA1"
        # board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)

        params = BrainFlowInputParams()
        board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

        board.prepare_session()
        board.start_stream(streamer_params="streaming_board://225.1.1.1:6677")
        BoardShim.log_message(LogLevels.LEVEL_INFO,
                              'start sleeping in the main thread')
        time.sleep(10)
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()
        return data

    def send_data_to_server(self, data):
        # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
        print('Transposed Data From the Board')
        df = pd.DataFrame(np.transpose(data), columns=self.column_labels)

        data_json = df.to_json()

        # Define the API endpoint URL
        url = 'http://35.206.70.248:5000/eegrandomforestprediction'

        # Set the request headers
        headers = {'Content-Type': 'application/json'}

        # Send the POST request with the data in the request body
        response = requests.post(url, data=data_json, headers=headers)
        return response

    def bciConnectionController(self):
        try:
            BoardShim.enable_dev_board_logger()

            # format data cols.
            self.column_labels = []
            for num in range(32):
                self.column_labels.append("c"+str(num))

            # read eeg data from the board -- will start a bci session with your current board
            # allowing it to stream to BCI Gui App and collect 10 second data sample
            data = self.read_from_board()
            # Sends preprocessed data via http request to get a prediction
            server_response = self.send_data_to_server(data)

            if server_response.status_code == 200:  # if a prediction is returned
                print(server_response.json())
                # return prediction
            else:  # there was in error in getting a thought prediction response
                # return error
                print("nay")
        except Exception as e:
            print(e)
            print("Error Occured during EEG data collection or transmission")

        # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
        # DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
        # restored_data = DataFilter.read_file('test.csv')
        # restored_df = pd.DataFrame(np.transpose(restored_data))
        # print('Data From the File')
        # print(restored_df.head(10))


if __name__ == "__main__":
    bcicon = bciConnection()
    bcicon.bciConnectionController()
