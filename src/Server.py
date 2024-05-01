import torch
import torch.optim as optim

import time

import Config
import Model
import Communication
import Checkpointer
import Dataset
import main as mainModule # Todo: clean up

class Server:

    class ServerClient:
        def __init__(self, comm, addr, model):
            self.comm = comm
            self.name = str(addr)
            self.model = model
            self.hasReceivedModel = False

    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.checkpointer = Checkpointer.Checkpointer(**config["checkpointer"])

        self.classifier = Model.Classifier(**config["classifier"]).to(device)
        self.classifierOptimizer = getattr(optim, config["optimizer"]["name"])(self.classifier.trainableParameters(), **config["optimizer"]["settings"])

        self.dataset = Dataset.Dataset(**config["dataset"])

        self.communicationServer = Communication.Server()

        self.clients = []
        self.currentModel = Model.BYOL(**self.config["EMA"], **self.config["BYOL"])

    def bind(self, ip, port):

        self.communicationServer.bind(ip, port)

    def listenForClients(self):

        while True:
            comm, addr = self.communicationServer.tryAcceptClient()
            if comm is None:
                break
            else:
                print(f"Accepted client at {addr}")

                client = self.ServerClient(comm, addr, Model.BYOL(**self.config["EMA"], **self.config["BYOL"]))
                self.clients.append(client)
                self.updateClientCommunication(client)

    def updateClientCommunication(self, client):
        if client.comm.isDataReady():
            message = client.comm.receiveMessage()
            if message == "stop": # Clients wants to disconnect
                print(f"Closing client {client.name}")
                self.clients.remove(client)
                client.comm.sendMessage("stopAcknowledged")
                client.comm.close()
            elif message == "requestSend": # Should send current model to client
                print(f"Sending model to client {client.name}")
                client.comm.sendModel(self.currentModel)
            elif message == "update":
                print(f"Receiving model from client {client.name}")
                client.comm.receiveModel(client.model)
                client.hasReceivedModel = True
            else:
                print(f"Received unknown message {message} from client {client.name}")

    def run(self):

        shouldStop = False
        while not shouldStop:

            print("Waiting for updated models...")
            while True: # Communicate with clients until all clients have sent an updated model
                self.listenForClients()
                for client in self.clients:
                    self.updateClientCommunication(client)

                clientsWithModelCount = 0
                for client in self.clients:
                    if client.hasReceivedModel:
                        clientsWithModelCount += 1

                if len(self.clients) == 0:
                    print("Waiting for clients...")
                elif clientsWithModelCount == len(self.clients):
                    break

                time.sleep(1)

            print("Averaging")

            modelParameters = [list((client.model.parameters())) for client in self.clients]

            # Todo: should we average all parameters or only the online parameters?
            for idx, param in enumerate(self.currentModel.parameters()):
                param.data = torch.mean(torch.stack([modelParameters[modelIdx][idx].data for modelIdx in range(len(modelParameters))]), dim=0)

            # Clear the received models to indicate we processed them
            for client in self.clients:
                client.hasReceivedModel = False

            # Communicate with clients to send the averaged model
            for client in self.clients:
                self.updateClientCommunication(client)

            print("Saving model")
            self.checkpointer.saveCheckpoint(self.currentModel, None)

            self.classifier.copyEncoderFromBYOL(self.currentModel)
            for i in range(self.config["server"]["classifierTrainEpochs"]):
                mainModule.TrainClassifierEpoch(self.classifier, self.device, self.dataset, self.classifierOptimizer, self.checkpointer, -1, -1)
            mainModule.TestEpoch(self.classifier, self.device, self.dataset)

def main():
    config = Config.GetConfig()
    config["device"] = "cpu"

    torch.manual_seed(0)
    device = torch.device(config["device"])

    server = Server(device, config)
    server.bind("localhost", 1234)
    server.run()

if __name__ == "__main__":
    main()