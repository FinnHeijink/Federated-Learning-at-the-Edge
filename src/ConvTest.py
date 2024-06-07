import Dataset
import KRIAInterface
import Config

def main():
    config = Config.GetConfig()
    dataset = Dataset.Dataset(**config["dataset"])

    _, batch = next(enumerate(dataset.testingEnumeration()))

    out = KRIAInterface.FConv2D_3x3(batch)
    print(out)

if __name__ == "__main__":
    main()