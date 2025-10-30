#!/usr/bin/env python2.7

from naoqi import ALProxy

import socket
import json
# NAO robot settings
NAO_IP = "192.168.23.53"
PORT = 9559


def connect_to_nao():
    """Try to connect to the NAO robot"""
    try:
        # Try to create a connection to NAO
        test_proxy = ALProxy("ALBehaviorManager", NAO_IP, PORT)
        print ("Successfully connected to NAO!")
        return True
    except Exception as e:
        print ("Failed to connect to NAO:", str(e))
        return False


def nao_sit_down():
    # Behavior name
    behavior_name = "sitdown-b3bee7/Sit down"

    # Create proxy to ALBehaviorManager
    try:
        behavior_mng = ALProxy("ALBehaviorManager", NAO_IP, PORT)
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

def nao_stand_up():
    # Behavior name
    behavior_name = "standup-80f5e5/Stand up"

    # Create proxy to ALBehaviorManager
    try:
        behavior_mng = ALProxy("ALBehaviorManager", NAO_IP,PORT)
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

def main():
    print ("Starting NAO Connection Service...")
    print ("Connecting to NAO at", NAO_IP, ":", PORT)
    
    # Create a socket to listen for connection requests
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 5000))
    server.listen(1)
    
    print ("Service ready on port 5000")
    print ("Waiting for connection request from GUI...")
    
    while True:
        try:
            # Wait for GUI to send a request
            client, address = server.accept()
            print("Received connection from", address)
            data = client.recv(1024).strip().lower()
            print("Received command:", data)


            if data == "connect":
            # Try to connect to NAO
                success = connect_to_nao()
            elif data == "sit_down":
                success = nao_sit_down()
            elif data == "stand_up":
                success = nao_stand_up()
            else:
                print("Unknown command pressed: ", data)
                success = False
            # Send response back to GUI
            response = "SUCCESS" if success else "FAILED"
            
            client.send(response)
            client.close()
            print ("Sent response:", response)
            
        except KeyboardInterrupt:
            print ("\nShutting down service...")
            break
        except Exception as e:
            print ("Error:", str(e))
    
    server.close()

if __name__ == "__main__":
    main()

