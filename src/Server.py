import torch
import torch.optim as optim

import Config
import Model
import Communication
import Checkpointer
import Dataset
import main as mainModule # Todo: clean up

class Server:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.checkpointer = Checkpointer.Checkpointer(**config["checkpointer"])

        self.classifier = Model.Classifier(**config["classifier"]).to(device)
        self.classifierOptimizer = getattr(optim, config["optimizer"]["name"])(self.classifier.trainableParameters(), **config["optimizer"]["settings"])

        self.dataset = Dataset.Dataset(**config["dataset"])

        self.clients = []

        self.communicationServer = Communication.Server()

    def listenForClients(self, ip, port, count):

        self.communicationServer.bind(ip, port)

        print("Listening for clients...")
        for i in range(count):
            client, addr = self.communicationServer.acceptClient()
            self.clients.append(client)

            print(f"Accepted client at {addr}")

    def run(self):

        # Sent a global start model first
        print("Sending initial model")
        initialModel = Model.BYOL(**self.config["EMA"], **self.config["BYOL"])

        for client in self.clients:
            client.sendModel(initialModel)

        shouldStop = False
        while not shouldStop:

            models = []

            print("Loading models...")

            for idx, client in enumerate(self.clients):
                print(f"Loading model from client {idx}")
                model = Model.BYOL(**self.config["EMA"], **self.config["BYOL"]).to(self.device)
                client.receiveModel(model.state_dict())
                models.append(model)

            print("Averaging")
            averagedModel = Model.BYOL(**self.config["EMA"], **self.config["BYOL"]).to(self.device)

            # Todo: should we average all parameters or only the online parameters?
            for params, toAverageParamsList in zip(averagedModel.parameters(), *[model.parameters() for model in models]):
                params.data = torch.mean(torch.stack([x.data for x in toAverageParamsList]), dim=0)

            print("Sending models...")

            for idx, client in enumerate(self.clients):
                print(f"Sending model to client {idx}")
                client.sendModel(averagedModel)

            print("Saving model")
            self.checkpointer.saveCheckpoint(averagedModel, None)

            #self.classifier.copyEncoderFromBYOL(averagedModel)
            #mainModule.TrainClassifierEpoch(self.classifier, self.device, self.dataset, self.classifierOptimizer, self.checkpointer, -1, -1)
            #mainModule.TestEpoch(self.classifier, self.device, self.dataset)

        for client in self.clients:
            client.close()

def main():
    config = Config.GetConfig()

    torch.manual_seed(0)
    device = torch.device(config["device"])

    server = Server(device, config)
    server.listenForClients("localhost", 1234, 1)
    server.run()

if __name__ == "__main__":
    main()