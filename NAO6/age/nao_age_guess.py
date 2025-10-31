import os
import subprocess
import time

import paramiko
from naoqi import ALProxy

# --- Config ---
NAO_IP = "192.168.23.53"
PORT = 9559
pic_format = "jpg"
PYTHON3_PATH = r"C:\Bounty4\ageenv\Scripts\python.exe"
ESTIMATED_AGE_FILE = "estimated_age.txt"

# --- Setup proxies ---
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)


def capture_and_download_nao_photo(nao_ip, port, local_path, photo_name):
    """Capture photo from NAO and download it"""
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
        camera.setPictureFormat(pic_format)
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
    """Run the age estimation script"""
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
    """Read and speak the estimated age"""
    try:
        with open(ESTIMATED_AGE_FILE, "r") as f:
            age = f.read().strip()
            sentence = "I think your age is " + age + " years old."
            print("Saying:", sentence)
            tts.say(sentence)
    except Exception as e:
        print("Error reading age:", e)
        tts.say("Sorry, I could not read your age.")


def main():
    """Main function to execute age guessing behavior"""
    try:
        # Define runtime directories
        output_dir = "C:/Bounty4"
        photo_base_name = "image_from_nao"

        # Build filenames
        photo_filename = "nao_photo." + pic_format
        local_filename = photo_base_name + "." + pic_format
        local_path = os.path.join(output_dir, local_filename)

        # Capture photo
        if not capture_and_download_nao_photo(NAO_IP, PORT, local_path, photo_filename):
            tts.say("Sorry, I could not take your picture.")
            return

        # Estimate age
        if not run_age_estimation_script(local_path):
            print("Failed to run age estimation script.")
            return

        # Speak result
        speak_estimated_age()

    except Exception as e:
        print("An error occurred:", e)
        tts.say("Sorry, an error occurred during age guessing.")


if __name__ == "__main__":
    main()
