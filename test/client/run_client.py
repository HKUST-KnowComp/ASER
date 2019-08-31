from tqdm import tqdm
from aser.client import ASERClient


if __name__ == "__main__":
    client = ASERClient(port=8000)

    # s1 = 'they yucca look leggy'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry'
    s2 = 'I am in the kitchen'

    print("Event 2 struct: ")
    print(client.extract_eventualities_struct(s2))

    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    e1 = client.get_exact_match_event(event1)
    e2 = client.get_exact_match_event(event2)
    print("Event 1: ")
    print(e1)
    print("Event 2: ")
    print(e2)
    print("Relation: ")
    rel = client.get_exact_match_relation(e1, e2)
    print(rel)

    # print("Related events: ")
    # related_events = client.get_related_events(e1)
    # print(related_events)

