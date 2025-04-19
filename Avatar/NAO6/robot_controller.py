import os
import qi
from dotenv import load_dotenv


load_dotenv()
ROBOT_IP = os.getenv("ROBOT_IP")
ROBOT_PORT = int(os.getenv("ROBOT_PORT", 9559))


class RobotController:
    def __init__(self, ip=ROBOT_IP, port=ROBOT_PORT):
        self.ip = ip
        self.port = port

        try:
            self.session = qi.Session()
            self.session.connect("tcp://{}:{}".format(self.ip, self.port))

            self.tts = self.session.service("ALTextToSpeech")
            self.motion = self.session.service("ALMotion")
            self.audio_player = self.session.service("ALAudioPlayer")
        except Exception as e:
            print("Failed to connect to NAO6 services: {}".format(e))
            raise

    def speak(self, text):
        try:
            self.tts.say(text)
        except Exception as e:
            print("Text-to-speech error: {}".format(e))

    def play_audio(self, file_path):
        try:
            return self.audio_player.playFile(file_path)
        except Exception as e:
            print("Audio playback error: {}".format(e))
            return None

    def stop_audio(self, song_id):
        try:
            self.audio_player.stop(song_id)
        except Exception as e:
            print("Audio stop error: {}".format(e))

    def move_left(self, distance = 0.05):
        try:
            self.motion.moveTo(0, distance, 0)
        except Exception as e:
            print("Move left error: {}".format(e))

    def move_right(self, distance = 0.05):
        try:
            self.motion.moveTo(0, -distance, 0)
        except Exception as e:
            print("Move right error: {}".format(e))

    def stop_movement(self):
        try:
            self.motion.stopMove()
        except Exception as e:
            print("Stop movement error: {}".format(e))

    def raise_arms(self):
        try:
            names = ["LShoulderPitch", "RShoulderPitch"]
            angles = [0.5, 0.5]
            speed = 0.2
            self.motion.setAngles(names, angles, speed)
        except Exception as e:
            print("Raise arms error: {}".format(e))

    def swing_arms(self):
        try:
            names = ["LShoulderPitch", "RShoulderPitch"]
            angles_up = [0.4, 0.4]
            angles_down = [0.6, 0.6]
            speed = 0.2

            self.motion.setAngles(names, angles_up, speed)
            self.motion.waitUntilMoveIsFinished()

            self.motion.setAngles(names, angles_down, speed)
            self.motion.waitUntilMoveIsFinished()
        except Exception as e:
            print("Swing arms error: {}".format(e))
