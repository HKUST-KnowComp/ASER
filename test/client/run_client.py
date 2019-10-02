from aser.client import ASERClient
from pprint import pprint as print


if __name__ == "__main__":
    client = ASERClient(port=12000, port_out=12001)

    s1 = 'they yucca look leggy. they african violet refuse to flower'
    s2 = 'they african violet refuse to flower'
    # s1 = 'The dog barks loudly.'
    # s2 = 'I am in the kitchen.'

    event1 = client.extract_eventualities(s1, only_events=False)
    event2 = client.extract_eventualities(s2, only_events=False)
    print("Event 1: ")
    print(event1)
    print("Event 2: ")
    print(event2)


    event_list1 = client.extract_eventualities(s1, only_events=True)
    event_list2 = client.extract_eventualities(s2, only_events=True)
    print("Event list 1 (Only Events): ")
    print(event_list1)
    print("Event list 2 (Only Events): ")
    print(event_list2)
    print("Relation: ")
    rel = client.predict_relation(event_list1[0], event_list2[0])
    print(rel)

    print("Related events: ")
    related_events = client.fetch_related_events(event_list1[0])
    print(related_events)

