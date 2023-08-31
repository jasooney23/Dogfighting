'''START THIS SERVER BEFORE STARTING THE GAME.'''

import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 11225  # The port used by the server

def send(client, msg):
    client.send(str(msg).encode())
    client.recv(1024)

def recv(client):
    data = client.recv(1024).decode()
    client.send("acknowledge".encode())
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    print("Waiting for game to connect...")

    server.listen(1)
    client, address = server.accept()
    print("Game connected!")

    while True:
        numMessages = int(recv(client))
        for x in range(numMessages):
            data = recv(client)

        numMessages = 5
        send(client, numMessages)
        for x in range(numMessages):
            send(client, x)
