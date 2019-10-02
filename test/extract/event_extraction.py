from pprint import pprint
from aser.extract.event_extractor import EventualityExtractor
from aser.extract.utils import parse_sentense_with_stanford


if __name__ == "__main__":
    e_extractor = EventualityExtractor(
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        corenlp_port=11001)

    text = "The dog barks loudly. Because he is hungry."
    res1 = e_extractor.extract_eventualities(text, only_events=False, output_format="json")
    parsed_result = parse_sentense_with_stanford(text, e_extractor.corenlp_client)[0]
    res2 = e_extractor.extract_eventualities_from_parsed_result(
        parsed_result, only_events=False, output_format="json")
    print()

    print("=" * 10 + "`EventualityExtractor.extract_eventualities`" + "=" * 10)
    print("Input: ", text)
    print("Output: ")
    pprint(res1)

    print("\n")
    print("=" * 10 + "`EventualityExtractor.extract_eventualities_from_parsed_result`" + "=" * 10)
    print("Input: ")
    pprint(parsed_result)
    print("Output: ")
    pprint(res2)