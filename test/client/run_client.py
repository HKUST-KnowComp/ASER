from aser.client import ASERClient
from pprint import pprint as print


if __name__ == "__main__":
    client = ASERClient(port=20097, port_out=20098)

    # s1 = 'they yucca look leggy. they african violet refuse to flower'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry'
    s2 = 'Evert said'

    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    print("Event 1: ")
    print(event1)
    # print(event1[0][0].to_dict())
    print("Event 2: ")
    print(event2)
    print("Relation: ")
    rel = client.predict_relation(event1[0][0], event2[0][0])
    print(rel)
    #
    print("Related events: ")
    related_events = client.fetch_related_events(event1[0][0])
    print(related_events)

