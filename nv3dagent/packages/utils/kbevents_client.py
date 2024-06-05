import socket

REMOTE_IP = "10.63.182.169"
PORT = 5000


def send_kb_event(kb_event, ip = REMOTE_IP, port = PORT):
    """
        Sending keyboard event for triggering to the remote server.
        Can support also combined keyboard events in the format like: "shift+2"
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))  # Replace 'remote_machine_ip' with the remote IP

    # Replace 'a' with the key you want to send
    client.send(kb_event.encode())
    response = client.recv(1024)
    print(response.decode())

    client.close()


if __name__ == "__main__":
    send_kb_event("shift+1")