from tqdm import tqdm
from aser.client import ASERClient


if __name__ == "__main__":
    client = ASERClient(port=8000)
    for i in tqdm(range(10000000)):
        client.extract_eventualities("the dog barks")
        pass