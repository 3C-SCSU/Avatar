import time

import pandas as pd
import requests
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, LogLevels

number = 0


class bciConnection:
    def read_from_board(self):
        # print(BoardIds.CYTON_BOARD.value)

        # use synthetic board for demo
        # params = BrainFlowInputParams()
        # params.serial_port = "COM6"
        # board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)

        params = BrainFlowInputParams()
        board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

        board.prepare_session()
        board.start_stream(streamer_params="streaming_board://225.1.1.1:6677")
        BoardShim.log_message(LogLevels.LEVEL_INFO, "start sleeping in the main thread")
        time.sleep(10)
        data = board.get_board_data()
        board.stop_stream()
        board.release_session()
        return data

    def send_data_to_server(self):
        # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
        # df = pd.DataFrame(np.transpose(data), columns=self.column_labels)

        global number
        number += 1
        temp_df = pd.read_csv(
            f"C:/Users/8pp84/Downloads/Skool/CSCI_495/data_repo/real/{number}.csv"
        )
        df = temp_df.drop(["TimestampFormatted"], axis=1)
        print("Read Data From file")

        data_json = df.to_json()

        # Define the API endpoint URL
        url = "http://127.0.0.1:5000/eegrandomforestprediction"

        # Set the request headers
        headers = {"Content-Type": "application/json"}

        # Send the POST request with the data in the request body
        response = requests.post(url, data=data_json, headers=headers)
        return response

    def bciConnectionController(self):
        try:
            BoardShim.enable_dev_board_logger()

            # format data cols.
            self.column_labels = [
                "SampleIndex",
                "EXGChannel0",
                "EXGChannel1",
                "EXGChannel2",
                "EXGChannel3",
                "EXGChannel4",
                "EXGChannel5",
                "EXGChannel6",
                "EXGChannel7",
                "EXGChannel8",
                "EXGChannel9",
                "EXGChannel10",
                "EXGChannel11",
                "EXGChannel12",
                "EXGChannel13",
                "EXGChannel14",
                "EXGChannel15",
                "AccelChannel0",
                "AccelChannel1",
                "AccelChannel2",
                "Other0",
                "Other1",
                "Other2",
                "Other3",
                "Other4",
                "Other5",
                "Other6",
                "AnalogChannel0",
                "AnalogChannel1",
                "AnalogChannel2",
                "Timestamp",
                "Other7",
            ]

            # read eeg data from the board -- will start a bci session with your current board
            # allowing it to stream to BCI Gui App and collect 10 second data sample
            # data = self.read_from_board()

            # Sends preprocessed data via http request to get a prediction
            server_response = self.send_data_to_server()

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
    bcicon = bciConnection()
    bcicon.bciConnectionController()
