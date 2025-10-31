from naoqi import ALProxy

# --- Config ---
NAO_IP = "192.168.23.53"
PORT = 9559

# --- Setup proxies ---
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)


def do_gangnam_style(nao_ip, port):
    """Execute Gangnam Style behavior"""
    behavior_name = "gangamstyle-d3f460/GangamStyle"

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
            import time

            while behavior_mng.isBehaviorRunning(behavior_name):
                time.sleep(1)

            return True
    else:
        print("Behavior not found on robot.")
        return False


def main():
    """Main function to execute Gangnam Style behavior"""
    try:
        if not do_gangnam_style(NAO_IP, PORT):
            tts.say("Sorry, I could not do Gangnam style.")
        else:
            print("Gangnam Style completed successfully!")
    except Exception as e:
        print("An error occurred:", e)
        tts.say("Sorry, an error occurred during Gangnam Style.")


if __name__ == "__main__":
    main()
