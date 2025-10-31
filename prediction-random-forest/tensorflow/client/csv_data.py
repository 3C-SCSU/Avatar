import pandas as pd
import requests


def send_data_to_server(data):
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    # df = pd.DataFrame(np.transpose(data), columns=self.column_labels)
    print(data.head(3))
    data_json = data.to_json()

    # Define the API endpoint URL
    url = "http://35.206.70.248:5000/eegrandomforestprediction"

    # Set the request headers
    headers = {"Content-Type": "application/json"}

    # Send the POST request with the data in the request body
    response = requests.post(url, data=data_json, headers=headers)

    return response


def get_last_prediction():
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    # df = pd.DataFrame(np.transpose(data), columns=self.column_labels)

    # Define the API endpoint URL
    url = "http://35.206.70.248:5000/lastprediction"

    # Set the request headers

    # Send the POST request with the data in the request body
    response = requests.get(url)

    return response


response = get_last_prediction()
print(response.json())

column_labels = []
for num in range(32):
    column_labels.append("c" + str(num))

df = pd.read_csv(
    "/Users/williamdoyle/Documents/GitHub/Brain-Computer-Interface/BrainFlow-RAW_Recordings_jn_land_5.csv",
    delimiter="\t",
    names=column_labels,
)

response = send_data_to_server(df)
print(response.json())
