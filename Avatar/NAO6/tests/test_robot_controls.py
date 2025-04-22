import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot_controller import RobotController


def test_robot_controller_initialization(mock_session_services):
    controller = RobotController()

    assert controller.session is mock_session_services["session"]
    assert controller.tts is mock_session_services["tts"]
    assert controller.motion is mock_session_services["motion"]
    assert controller.audio_player is mock_session_services["audio_player"]


def test_speak(mock_session_services):
    controller = RobotController()
    controller.speak("Hello world")
    mock_session_services["tts"].say.assert_called_once_with("Hello world")


def test_play_audio(mock_session_services):
    controller = RobotController()
    controller.play_audio("/path/to/audio.mp3")
    mock_session_services["audio_player"].playFile.assert_called_once_with("/path/to/audio.mp3")


def test_stop_audio(mock_session_services):
    controller = RobotController()
    controller.stop_audio(42)
    mock_session_services["audio_player"].stop.assert_called_once_with(42)


def test_move_left(mock_session_services):
    controller = RobotController()
    controller.move_left()
    mock_session_services["motion"].moveTo.assert_called_once_with(0, 0.05, 0)


def test_move_right(mock_session_services):
    controller = RobotController()
    controller.move_right()
    mock_session_services["motion"].moveTo.assert_called_once_with(0, -0.05, 0)


def test_stop_movement(mock_session_services):
    controller = RobotController()
    controller.stop_movement()
    mock_session_services["motion"].stopMove.assert_called_once()


def test_raise_arms(mock_session_services):
    controller = RobotController()
    controller.raise_arms()
    mock_session_services["motion"].setAngles.assert_called_once_with(
        ["LShoulderPitch", "RShoulderPitch"], [0.5, 0.5], 0.2
    )


def test_swing_arms(mock_session_services):
    controller = RobotController()
    controller.swing_arms()
    assert mock_session_services["motion"].setAngles.call_count == 2
    mock_session_services["motion"].waitUntilMoveIsFinished.assert_called()
