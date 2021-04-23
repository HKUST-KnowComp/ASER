from aser.client import ASERClient
from pprint import pprint as print


if __name__ == "__main__":
    client = ASERClient(port=20097, port_out=20098)

    # s1 = 'they yucca look leggy. they african violet refuse to flower'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry'
    s2 = 'Evert said'

    eventuality1 = client.extract_eventualities(s1)
    eventuality2 = client.extract_eventualities(s2)
    print("Event 1: ")
    print(eventuality1)
    # print(eventuality1[0][0].to_dict())
    print("Event 2: ")
    print(eventuality2)
    print("Relation: ")
    rel = client.predict_relation(eventuality1[0][0], eventuality2[0][0])
    print(rel)
    #
    print("Related events: ")
    related_events = client.fetch_related_eventualities(eventuality1[0][0])
    print(related_events)

