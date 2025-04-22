import time
from robot_controller import RobotController


SONG_PATH = "/home/nao/music/gangnam_style.mp3"  # song should be in NAO robot's system


def main() -> None:
    robot = RobotController()
    robot.speak("All right!")
    song_id = robot.play_audio(SONG_PATH)

    dance_time = 30
    start_time = time.time()
    while time.time() - start_time < dance_time:
        robot.move_left()
        robot.raise_arms()
        time.sleep(0.5)
        robot.move_right()
        robot.swing_arms()
        time.sleep(0.5)

    if song_id:
        robot.stop_audio(song_id)
    robot.stop_movement()
    robot.speak("Hope you enjoyed it!")


if __name__ == "__main__":
    main()
