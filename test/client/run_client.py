from tqdm import tqdm
from aser.client import ASERClient
from pprint import pprint as print


if __name__ == "__main__":
    client = ASERClient(port=8000, port_out=8001)

    # s1 = 'they yucca look leggy'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry.'
    s2 = 'I am in the kitchen.'

    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    print("Event 1: ")
    print(event1)
    print("Event 2: ")
    print(event2)


    event1 = client.extract_eventualities(s1, only_events=True)[0]
    event2 = client.extract_eventualities(s2, only_events=True)[0]
    print("Relation: ")
    rel = client.predict_relation(event1, event2)
    print(rel)

    print("Related events: ")
    related_events = client.fetch_related_events(event2)
    print(related_events)

