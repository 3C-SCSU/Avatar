import sys
import time

from naoqi import ALProxy

# --- Config ---
NAO_IP = "192.168.23.53"
PORT = 9559

# --- Setup proxies ---
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)


def nao_sit_down(nao_ip, port):
    """Execute sit down behavior"""
    behavior_name = "sitdown-b3bee7/Sit down"

    try:
        behavior_mng = ALProxy("ALBehaviorManager", nao_ip, port)
    except Exception as e:
        print("Could not create proxy to ALBehaviorManager")
        print("Error:", e)
        return False

    # Check if behavior is installed
    if behavior_mng.isBehaviorInstalled(behavior_name):
        if behavior_mng.isBehaviorRunning(behavior_name):
            print("Behavior is already running.")
            return True
        else:
            print("Starting behavior:", behavior_name)
            behavior_mng.startBehavior(behavior_name)

            # Wait for behavior to complete
            while behavior_mng.isBehaviorRunning(behavior_name):
                time.sleep(1)

            return True
    else:
        print("Behavior not found on robot.")
        return False


def nao_stand_up(nao_ip, port):
    """Execute stand up behavior"""
    behavior_name = "standup-80f5e5/Stand up"

    try:
        behavior_mng = ALProxy("ALBehaviorManager", nao_ip, port)
    except Exception as e:
        print("Could not create proxy to ALBehaviorManager")
        print("Error:", e)
        return False

    # Check if behavior is installed
    if behavior_mng.isBehaviorInstalled(behavior_name):
        if behavior_mng.isBehaviorRunning(behavior_name):
            print("Behavior is already running.")
            return True
        else:
            print("Starting behavior:", behavior_name)
            behavior_mng.startBehavior(behavior_name)

            # Wait for behavior to complete
            while behavior_mng.isBehaviorRunning(behavior_name):
                time.sleep(1)

            return True
    else:
        print("Behavior not found on robot.")
        return False


def main():
    """Main function to execute sit/stand behavior"""
    if len(sys.argv) < 2:
        print("Usage: python nao_sit_stand.py [sit|stand]")
        return

    action = sys.argv[1].lower()

    try:
        if action == "sit":
            if not nao_sit_down(NAO_IP, PORT):
                tts.say("Sorry, I could not sit down, I have pain in my knees.")
            else:
                print("Sit down completed successfully!")
        elif action == "stand":
            if not nao_stand_up(NAO_IP, PORT):
                tts.say("Sorry, I could not stand up, because I'm lazy.")
            else:
                print("Stand up completed successfully!")
        else:
            print("Invalid action. Use 'sit' or 'stand'.")
    except Exception as e:
        print("An error occurred:", e)
        tts.say("Sorry, an error occurred during the movement.")


if __name__ == "__main__":
    main()
