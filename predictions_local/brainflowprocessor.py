import time
import pandas as pd
import torch
import serial # required pyserial install
import brainflow
from sklearn.preprocessing import StandardScaler
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from serial.tools import list_ports

class BrainFlowDataProcessor:
    """
    Handles EEG data ingestion from BrainFlow boards (Synthetic, Cyton, or Cyton Daisy),
    labeling channels, extracting EEG signals, and preparing preprocessed tensors.
    """

    CYTON_BOARDS = [BoardIds.CYTON_BOARD.value, BoardIds.CYTON_DAISY_BOARD.value]

    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD.value, sample_time=2):
        self.board_id = board_id
        self.sample_time = sample_time
        self.params = BrainFlowInputParams()
        
        # Auto-detect serial port only for Cyton-type boards
        if self.board_id in self.CYTON_BOARDS:
            self._auto_set_serial_port()
        
        self.board = None
        self.df = None
        self.eeg_cols = None
        self.X_tensor = None

    def _auto_set_serial_port(self):
        # Auto-detect Cyton serial port using pyserial
        ports = list_ports.comports()
        detected_port = None
        for p in ports:
            desc = p.description.lower()
            hwid = p.hwid.lower()
            if "usb" in desc or "ftdi" in desc or "serial" in desc:
                detected_port = p.device
                break

        if detected_port is None:
            raise RuntimeError("Could not auto-detect Cyton serial port. Please specify manually.")
        self.params.serial_port = detected_port
        print(f"[INFO] Auto-detected Cyton serial port: {detected_port}")

    def _connect_board(self):
        # Prepare and start board session
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.board.start_stream()
        time.sleep(self.sample_time)

    def _disconnect_board(self):
        # Stop streaming and release board session
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
            self.board = None

    def capture_data(self):
        # Capture raw data from the board and store as pandas DataFrame
        try:
            self._connect_board()
            data = self.board.get_board_data()
        finally:
            self._disconnect_board()

        descr = BoardShim.get_board_descr(self.board_id)
        channel_names = [""] * data.shape[0]

        # Map channel indices to names
        if 'package_num_channel' in descr:
            channel_names[descr['package_num_channel']] = "PUC"
        for idx in descr.get('eeg_channels', []):
            channel_names[idx] = f"EEG_{idx}"
        for idx in descr.get('accel_channels', []):
            channel_names[idx] = f"ACCEL_{idx}"
        for idx in descr.get('gyro_channels', []):
            channel_names[idx] = f"GYRO_{idx}"
        for idx in descr.get('eda_channels', []):
            channel_names[idx] = f"EDA_{idx}"
        for idx in descr.get('ppg_channels', []):
            channel_names[idx] = f"PPG_{idx}"
        for idx in descr.get('temperature_channels', []):
            channel_names[idx] = f"TEMPERATURE_{idx}"
        for idx in descr.get('resistance_channels', []):
            channel_names[idx] = f"RESISTANCE_{idx}"
        if 'battery_channel' in descr:
            channel_names[descr['battery_channel']] = "BATTERY"
        if 'timestamp_channel' in descr:
            channel_names[descr['timestamp_channel']] = "TIMESTAMP"
        if 'marker_channel' in descr:
            channel_names[descr['marker_channel']] = "MARKER"

        # Put into DataFrame (transpose so rows = samples)
        self.df = pd.DataFrame(data.T, columns=channel_names) 
        return self.df

    def extract_eeg_data(self):
        # Extract EEG columns only
        if self.df is None:
            raise ValueError("Data not captured yet. Call capture_data() first.")
        self.eeg_cols = [c for c in self.df.columns if c.startswith("EEG_")]
        self.eeg_df = self.df[self.eeg_cols]
        return self.eeg_df

    def preprocess_eeg(self):
        # Standardize EEG signals and convert to PyTorch tensor
        if self.eeg_df is None:
            self.extract_eeg_data()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.eeg_df)
        self.X_tensor = torch.tensor(X_scaled, dtype=torch.float)
        return self.X_tensor
   
    def get_tensor(self):
        # Capture, extract, and preprocess EEG in one call
        self.capture_data()
        self.extract_eeg_data()
        return self.preprocess_eeg()
    

if __name__ == "__main__":
    BrainFlowDataProcessor()