import time
import numpy as np
import pandas as pd
import requests
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from enum import Enum

class DataMode(Enum):
    SYNTHETIC = "Synthetic"
    LIVE = "Live"


class bciConnection():

    __instance = None

    def __init__(self, serial_port : str = "/dev/cu.usbserial-D200PMA1", mode: DataMode = DataMode.SYNTHETIC):
        if bciConnection.__instance != None:
            raise Exception("This should be a singleton class")
        
        bciConnection.__instance = self
        self.__params = BrainFlowInputParams()
        self.__serial_port = serial_port
        self.__mode = mode

    @staticmethod
    def get_instance(serial_port : str = "/dev/cu.usbserial-D200PMA1",  mode: DataMode = DataMode.SYNTHETIC):
        if bciConnection.__instance == None:
            return bciConnection(serial_port, mode)
        return bciConnection.__instance

    def set_mode(self, mode: DataMode):
        self.__mode = mode

    def read_from_board(self):

        if self.__mode == DataMode.LIVE:
            self.__params.serial_port = self.__serial_port
            board = BoardShim(BoardIds.CYTON_BOARD.value, self.__params)
        else:
            board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, self.__params)

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
        url = 'http://127.0.0.1:5000/eegrandomforestprediction'

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
                return server_response.json()
                # return prediction
            else:  # there was in error in getting a thought prediction response
                # return error
                print(server_response.status_code)
        except Exception as e:
            print(e)
            print(server_response.status_code)
            print("Error Occured during EEG data collection or transmission")

        # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
        # DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
        # restored_data = DataFilter.read_file('test.csv')
        # restored_df = pd.DataFrame(np.transpose(restored_data))
        # print('Data From the File')
        # print(restored_df.head(10))


if __name__ == "__main__":
    bcicon = bciConnection.get_instance()
    bcicon.bciConnectionController()
