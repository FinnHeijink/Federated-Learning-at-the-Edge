import socket

class Communication:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, port):
        self.socket.connect(('localhost', port))

    def close(self):
        self.socket.close()

    def sendModel(self, stateDict):
        pass

    def receiveModel(self, stateDict):
        pass
