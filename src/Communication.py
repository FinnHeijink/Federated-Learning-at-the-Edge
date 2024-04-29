import socket
import torch
import struct
import io

class Communication:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, port):
        self.socket.connect(('localhost', port))

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
        length = struct.unpack("!i", packedLength)

        data = self.socket.recv(length)

        bytesReadStream = io.BytesIO(data)
        torch.load(bytesReadStream, stateDict)
