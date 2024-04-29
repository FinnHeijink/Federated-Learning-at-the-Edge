import socket
import torch
import struct
import io

class Server:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def bind(self, ip, port):
        self.socket.bind((ip, port))

    def close(self):
        self.socket.close()

    def acceptClient(self):
        self.socket.listen()
        clientSocket, addr = self.socket.accept()
        return Communication(clientSocket), addr

class Communication:
    def __init__(self, initialSocket = None):
        if initialSocket == None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.socket = initialSocket

    def connect(self, ip, port):
        self.socket.connect((ip, port))

    def close(self):
        self.socket.close()

    def sendModel(self, stateDict):
        bytesStream = io.BytesIO()
        torch.save(stateDict, bytesStream)

        data = bytesStream.getvalue()
        length = len(data)

        packedLength = struct.pack("!i", length)

        self.socket.send(packedLength)
        self.socket.send(data)

    def receiveModel(self, stateDict):
        packedLength = self.socket.recv(4)
        length = struct.unpack("!i", packedLength)[0]

        data = self.socket.recv(length)

        bytesReadStream = io.BytesIO(data)
        torch.load(bytesReadStream, stateDict)
