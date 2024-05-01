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

    def sendModel(self, model):
        bytesStream = io.BytesIO()
        torch.save(model.state_dict(), bytesStream)

        data = bytesStream.getvalue()
        length = len(data)

        packedLength = struct.pack("!i", length)

        self.socket.sendall(packedLength)
        self.socket.sendall(data)

    def receiveModel(self, model):
        packedLength = recvall(self.socket, 4)
        length = struct.unpack("!i", packedLength)[0]

        data = recvall(self.socket, length)

        bytesReadStream = io.BytesIO(data)
        statesDict = model.state_dict()
        model.load_state_dict(torch.load(bytesReadStream))
        bytesReadStream.close()

def recvall(socket, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data