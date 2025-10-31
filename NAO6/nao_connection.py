import socket

NAO_IP = "192.168.23.53"
PORT = 9559


def send_command(command):
    """
    Sends a command to the Python 2.7 NAO service.
    Supported commands: 'connect', 'sit_down', 'stand_up'
    """
    try:
        print(f"[DEBUG] Attempting to send command: {command}")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5.0)  # Add timeout
        client.connect(("localhost", 5000))
        print("[DEBUG] Connected to service")

        client.send(command.encode())
        print("[DEBUG] Command sent")

        response = client.recv(1024).decode().strip()
        print(f"[DEBUG] Received response: {response}")
        client.close()
        return response

    except socket.timeout:
        print("[ERROR] Connection timeout - is nao_service.py running?")
        return None
    except ConnectionRefusedError:
        print("[ERROR] Connection refused - nao_service.py is NOT running on port 5000")
        return None
    except Exception as e:
        print(f"[ERROR] Error sending command '{command}': {e}")
        return None
