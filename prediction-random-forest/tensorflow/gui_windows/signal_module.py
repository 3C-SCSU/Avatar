import threading
import time


class signalling_system:
    """
    This class intends to send the drone a non-movement signal every 2 seconds

    Attributes:
        state (bool): Mimics the drone state.
        callback (function): External function the signalling_system use to update drone.
        signal (str): The message the signalling_system sends to the drone
        generator_thread (threading.Thread): The thread responsible for generating signal.

    Methods:
        trigger(): Sends out a signal approximately every 10 seconds while drone is left turned on.
        start(): Starts the signalling thread.
        stop(): Stops the signalling thread.
    """

    def __init__(self, callback, signal):
        self.state = False
        self.callback = callback
        self.signal = signal
        self.generator_thread = threading.Thread(target=self.trigger)

    def trigger(self):
        # While Drone is active (user has not turned it off manually).
        while self.state:
            # Wait for 2 seconds
            time.sleep(2)

            # Check the state before sending the signal
            if self.state:
                self.callback(self.signal)

    def start(self):
        self.state = True
        self.generator_thread.start()

    def stop(self):
        self.state = False
        self.generator_thread.join()
