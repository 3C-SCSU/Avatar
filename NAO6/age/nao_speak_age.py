from naoqi import ALProxy
import time
import subprocess
import paramiko
import os

# --- Config ---
NAO_IP = "192.168.43.100"
PORT = 9559
pic_format = "jpg"
PYTHON3_PATH = r"C:\Bounty4\ageenv\Scripts\python.exe"
ESTIMATED_AGE_FILE = "estimated_age.txt"

# --- Setup global proxies ---
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)
asr = ALProxy("ALSpeechRecognition", NAO_IP, PORT)
memory = ALProxy("ALMemory", NAO_IP, PORT)
manager = ALProxy("ALBehaviorManager", NAO_IP, PORT)

def setup_speech_recognition():
    # Unsubscribe from all previous apps
    for subscriber in ["GuessAgeApp", "TestASR", "ASRApp", "SpeechApp"]:
        try:
            asr.unsubscribe(subscriber)
        except:
            pass

    try:
        asr.pause(True)
    except:
        pass

    try:
        memory.removeData("WordRecognized")
    except:
        pass

    manager.stopAllBehaviors()

    asr.pause(True)
    asr.setLanguage("English")
    asr.setAudioExpression(False)
    asr.setVocabulary(["Do Gangnam Style", "NAO guess my age", "Stand Up", "Sit Down"], False)
    tts.say("I'm listening. Say")
    asr.pause(False)
    try:
        memory.removeData("WordRecognized")
    except:
        pass

    asr.subscribe("GuessAgeApp")
    time.sleep(2)


def setup_exit_continue(activity):
    # Unsubscribe from all previous apps
    for subscriber in ["GuessAgeApp", "TestASR", "ASRApp", "SpeechApp"]:
        try:
            asr.unsubscribe(subscriber)
        except:
            pass

    # manager.stopAllBehaviors()
    asr.pause(True)
    asr.setLanguage("English")
    asr.setAudioExpression(False)
    asr.setVocabulary(["Yes", "No"], False)
    if activity == "Gangnam":
        time.sleep(20)
    elif activity == "Age":
        time.sleep(2)
    else:
        time.sleep(15)
    tts.say("Do you want to see anything else? Say Yes or NO")
    asr.pause(False)
    try:
        memory.removeData("WordRecognized")
    except:
        pass

    asr.subscribe("GuessAgeApp")
    time.sleep(2)
    last_word = ""
    exit_flag = False
    try:
        val = memory.getData("WordRecognized")
        if isinstance(val, list) and len(val) >= 2:
            current_word = val[0]
            confidence = val[1]
            if current_word != last_word:
                print("Heard:", current_word, "| Confidence:", confidence)

            if current_word == "Yes" and confidence > 0.4:
                tts.say("Sure, what else do you want me to do?")
                exit_flag = False

            elif current_word == "No" and confidence > 0.4:
                exit_flag = True

    except RuntimeError:
        print("No word was recognized.")
        tts.say("I did not hear anything.")
        exit_flag = True  # or False based on your design

    asr.unsubscribe("GuessAgeApp")
    return exit_flag

    

def wait_for_phrase(max_attempts):
    activity = ""
    attempts = 0
    last_word = ""
    while attempts < max_attempts:
        try:
            data = memory.getData("WordRecognized")
            if isinstance(data, list) and len(data) >= 2:
                current_word = data[0]
                confidence = data[1]
                if current_word != last_word:
                    print("Heard:", current_word, "| Confidence:", confidence)
                attempts += 1

                if current_word == "NAO guess my age" and confidence > 0.3:
                    tts.say("Okay, let me guess your age.")
                    asr.unsubscribe("GuessAgeApp")
                    activity = "Age"
                    return True, activity

                elif current_word == "Do Gangnam Style" and confidence > 0.3:
                    tts.say("Okay, let me do Gangnam Style.")
                    asr.unsubscribe("GuessAgeApp")
                    activity = "Gangnam"
                    return True, activity
                
                elif current_word == "Stand Up" and confidence > 0.3:
                    tts.say("Okay, let me Stand Up.")
                    asr.unsubscribe("GuessAgeApp")
                    activity = "standup"
                    return True, activity
                
                elif current_word == "Sit Down" and confidence > 0.3:
                    tts.say("Okay, let me Sit Down")
                    asr.unsubscribe("GuessAgeApp")
                    activity = "sitdown"
                    return True, activity
                
                last_word = current_word
        except:
            pass
        time.sleep(0.3)

    if attempts >= max_attempts:
        tts.say("I did not understand. Please try again.")
    asr.unsubscribe("GuessAgeApp")
    return False, activity

def capture_and_download_nao_photo(nao_ip, port, local_path, photo_name):
    photo_dir = "/home/nao/recordings/cameras/"
    full_nao_path = photo_dir + photo_name

    # Move head to look straight
    motion = ALProxy("ALMotion", nao_ip, port)
    motion.setStiffnesses("Head", 1.0)
    motion.setAngles(["HeadPitch", "HeadYaw"], [0.0, 0.0], 0.2)

    try:
        for count in ["3", "2", "1", "Cheese"]:
            tts.say(count)
            time.sleep(0.8)
    except Exception as e:
        print("Countdown failed:", e)
        tts.say("Something went wrong with the countdown.")

    try:
        camera = ALProxy("ALPhotoCapture", nao_ip, port)       
        camera.setResolution(2)
        camera.setPictureFormat(pic_format)  # use global format
        motion.setStiffnesses("Head", 1.0)
        motion.setAngles(["HeadPitch", "HeadYaw"], [0.0, 0.0], 0.2)
        camera.takePicture(photo_dir, photo_name.split(".")[0])
        print("NAO took a picture!")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(nao_ip, username="nao", password="nao")

        sftp = ssh.open_sftp()
        sftp.get(full_nao_path, local_path)
        print("Photo downloaded to:", local_path)

        sftp.remove(full_nao_path)
        print("Photo deleted from NAO storage")

        sftp.close()
        ssh.close()
    except Exception as e:
        print("SFTP error:", e)
        return False

    return True

def run_age_estimation_script(image_path):
    try:
        print("Running Python 3 script to estimate age...")
        subprocess.call([PYTHON3_PATH, "age_guess.py", image_path])
        time.sleep(1)
    except Exception as e:
        print("Failed to run Python 3 script:", e)
        tts.say("Sorry, I could not process your age.")
        return False
    return True

def speak_estimated_age():
    try:
        with open(ESTIMATED_AGE_FILE, "r") as f:
            age = f.read().strip()
            sentence = "I think your age is " + age + " years old."
            print("Saying:", sentence)
            tts.say(sentence)
    except Exception as e:
        print("Error reading age:", e)
        tts.say("Sorry, I could not read your age.")

def do_gangnam_style(nao_ip, port):
    # Behavior name
    behavior_name = "gangamstyle-d3f460/GangamStyle"

    # Create proxy to ALBehaviorManager
    try:
        behavior_mng = ALProxy("ALBehaviorManager", nao_ip, port)
    except Exception as e:
        print("Could not create proxy to ALBehaviorManager")
        print("Error:", e)
        exit(1)

    # Check if behavior is installed
    if behavior_mng.isBehaviorInstalled(behavior_name):
        if behavior_mng.isBehaviorRunning(behavior_name):
            print("Behavior is already running.")
            return True
        else:
            print("Starting behavior:", behavior_name)
            behavior_mng.startBehavior(behavior_name)
            return True
    else:
        print("Behavior not found on robot.")

    return False


def nao_sit_down(nao_ip, port):
    # Behavior name
    behavior_name = "sitdown-b3bee7/Sit down"

    # Create proxy to ALBehaviorManager
    try:
        behavior_mng = ALProxy("ALBehaviorManager", nao_ip, port)
    except Exception as e:
        print("Could not create proxy to ALBehaviorManager")
        print("Error:", e)
        exit(1)

    # Check if behavior is installed
    if behavior_mng.isBehaviorInstalled(behavior_name):
        if behavior_mng.isBehaviorRunning(behavior_name):
            print("Behavior is already running.")
            return True
        else:
            print("Starting behavior:", behavior_name)
            behavior_mng.startBehavior(behavior_name)
            return True
    else:
        print("Behavior not found on robot.")

    return False

def nao_stand_up(nao_ip, port):
    # Behavior name
    behavior_name = "standup-80f5e5/Stand up"

    # Create proxy to ALBehaviorManager
    try:
        behavior_mng = ALProxy("ALBehaviorManager", nao_ip, port)
    except Exception as e:
        print("Could not create proxy to ALBehaviorManager")
        print("Error:", e)
        exit(1)

    # Check if behavior is installed
    if behavior_mng.isBehaviorInstalled(behavior_name):
        if behavior_mng.isBehaviorRunning(behavior_name):
            print("Behavior is already running.")
            return True
        else:
            print("Starting behavior:", behavior_name)
            behavior_mng.startBehavior(behavior_name)
            return True
    else:
        print("Behavior not found on robot.")

    return False

def choose_activity(NAO_IP, PORT, local_path, photo_filename, activity):
    if activity == "Age":
        if not capture_and_download_nao_photo(NAO_IP, PORT, local_path, photo_filename):
            tts.say("Sorry, I could not take your picture.")
            return
        else:
            if not run_age_estimation_script(local_path):
                print("Failed to run age estimation script.")

            speak_estimated_age()
            tts.say("Thank you for your time. Goodbye!")
            return
        return
    elif activity == "Gangnam":
        if not do_gangnam_style(NAO_IP, PORT):
            tts.say("Sorry, I could not do Gangnam style.")
        
        return
    elif activity == "sitdown":
        if not nao_sit_down(NAO_IP, PORT):
            tts.say("Sorry, I could not do sit down, I have pain in my knees.")
        return
    
    elif activity == "standup":
        if not nao_stand_up(NAO_IP, PORT):
            tts.say("Sorry, I could not do stand up, because I'm lazy.")
        return

def main():
    try:
        # Define runtime directories
        output_dir = "C:/Bounty4"
        photo_base_name = "image_from_nao"
        success = False
        activity = ""
        activity_flag = True

        # Build filenames using global pic_format
        photo_filename = "nao_photo." + pic_format
        local_filename = photo_base_name + "." + pic_format
        local_path = os.path.join(output_dir, local_filename)
        
    
        # Start voice interaction
        while activity_flag:
            setup_speech_recognition()
            success, activity = wait_for_phrase(max_attempts=5)
            if success:
                choose_activity(NAO_IP, PORT, local_path, photo_filename, activity)
                if not setup_exit_continue(activity):
                    continue
                else:
                    tts.say("Thank you for your time. Goodbye!")
                    break  
            else:
                tts.say("Let's try that again.")
                time.sleep(1)

    except Exception as e:
        print("An error occurred:", e)
        tts.say("Sorry, an error occurred.")


        # if activity == "Age":
        #     if not capture_and_download_nao_photo(NAO_IP, PORT, local_path, photo_filename):
        #         tts.say("Sorry, I could not take your picture.")
        #         return
        #     else:
        #         if not run_age_estimation_script(local_path):
        #             print("Failed to run age estimation script.")

        #         speak_estimated_age()
        #         tts.say("Thank you for your time. Goodbye!")
        #         return
        # elif activity == "Gangnam":
        #     if not do_gangnam_style(NAO_IP, PORT):
        #         tts.say("Sorry, I could not do Gangnam style.")
            
        #     return
        # elif activity == "sitdown":
        #     if not nao_sit_down(NAO_IP, PORT):
        #         tts.say("Sorry, I could not do sit down, I have pain in my knees.")
        #     return
        
        # elif activity == "standup":
        #     if not nao_stand_up(NAO_IP, PORT):
        #         tts.say("Sorry, I could not do stand up, because I'm lazy.")
        #     return

if __name__ == "__main__":
    main()