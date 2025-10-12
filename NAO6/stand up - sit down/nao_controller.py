from naoqi import ALProxy
import time
import subprocess

# --- Config ---
NAO_IP = "192.168.23.53"
PORT = 9559

# --- Setup global proxies ---
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)
asr = ALProxy("ALSpeechRecognition", NAO_IP, PORT)
memory = ALProxy("ALMemory", NAO_IP, PORT)
manager = ALProxy("ALBehaviorManager", NAO_IP, PORT)

def setup_speech_recognition():
    """Initialize speech recognition with available commands"""
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
    tts.say("I'm listening. Say a command.")
    asr.pause(False)
    try:
        memory.removeData("WordRecognized")
    except:
        pass

    asr.subscribe("GuessAgeApp")
    time.sleep(2)


def setup_exit_continue():
    """Ask user if they want to continue or exit"""
    # Unsubscribe from all previous apps
    for subscriber in ["GuessAgeApp", "TestASR", "ASRApp", "SpeechApp"]:
        try:
            asr.unsubscribe(subscriber)
        except:
            pass

    asr.pause(True)
    asr.setLanguage("English")
    asr.setAudioExpression(False)
    asr.setVocabulary(["Yes", "No"], False)
    time.sleep(2)
    tts.say("Do you want to see anything else? Say Yes or No")
    asr.pause(False)
    try:
        memory.removeData("WordRecognized")
    except:
        pass

    asr.subscribe("GuessAgeApp")
    time.sleep(2)
    
    exit_flag = False
    attempts = 0
    max_attempts = 10
    
    while attempts < max_attempts:
        try:
            val = memory.getData("WordRecognized")
            if isinstance(val, list) and len(val) >= 2:
                current_word = val[0]
                confidence = val[1]
                print("Heard:", current_word, "| Confidence:", confidence)

                if current_word == "Yes" and confidence > 0.4:
                    tts.say("Sure, what else do you want me to do?")
                    exit_flag = False
                    break

                elif current_word == "No" and confidence > 0.4:
                    exit_flag = True
                    break
        except RuntimeError:
            pass
        
        attempts += 1
        time.sleep(0.3)
    
    if attempts >= max_attempts:
        print("No clear response received.")
        tts.say("I did not hear anything clearly.")
        exit_flag = True

    asr.unsubscribe("GuessAgeApp")
    return exit_flag

    

def wait_for_phrase(max_attempts):
    """Wait for user to say a command"""
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


def execute_activity(activity):
    """Execute the requested activity by calling appropriate script"""
    if activity == "Age":
        subprocess.call(["python", "nao_age_guess.py"])
    elif activity == "Gangnam":
        subprocess.call(["python", "nao_gangnam.py"])
    elif activity == "sitdown":
        subprocess.call(["python", "nao_sit_stand.py", "sit"])
    elif activity == "standup":
        subprocess.call(["python", "nao_sit_stand.py", "stand"])


def main():
    try:
        tts.say("Hello! I am NAO robot.")
        
        while True:
            setup_speech_recognition()
            success, activity = wait_for_phrase(max_attempts=10)
            
            if success:
                execute_activity(activity)
                
                if setup_exit_continue():
                    tts.say("Thank you for your time. Goodbye!")
                    break
            else:
                tts.say("Let's try that again.")
                time.sleep(1)

    except Exception as e:
        print("An error occurred:", e)
        tts.say("Sorry, an error occurred.")


if __name__ == "__main__":
    main()