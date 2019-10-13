from pprint import pprint
from aser.extract.event_extractor import EventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor
from aser.extract.utils import parse_sentense_with_stanford

if __name__ == "__main__":
    e_extractor = EventualityExtractor(
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        corenlp_port=13000)

    ##############################################################
    ##
    ## Extract eventualities from raw sentence
    ##
    ##############################################################

    text = "I go to lunch because I am hungry. I go to kitchen before I find some food"
    eventualities_list = e_extractor.extract_eventualities(text)

    print("=" * 10 + "`EventualityExtractor.extract_eventualities`" + "=" * 10)
    print("Input: ", text)
    print("Output: ")
    for elist in eventualities_list:
        print(elist)

    ##############################################################
    ##
    ## Extract eventualities from stanford corenlp parsed results
    ##
    ##############################################################

    print("\n")
    parsed_result = {'dependencies': [(1, 'nsubj', 0),
                                      (1, 'nmod:to', 3),
                                      (1, 'advcl:because', 7),
                                      (1, 'punct', 8),
                                      (3, 'case', 2),
                                      (7, 'mark', 4),
                                      (7, 'nsubj', 5),
                                      (7, 'cop', 6)],
                     'lemma': ['I', 'go', 'to', 'lunch', 'because', 'I', 'be', 'hungry', '.'],
                     'pos_tags': ['PRP', 'VBP', 'TO', 'NN', 'IN', 'PRP', 'VBP', 'JJ', '.'],
                     'tokens': ['I', 'go', 'to', 'lunch', 'because', 'I', 'am', 'hungry', '.']}

    eventualities = e_extractor.extract_eventualities_from_parsed_result(parsed_result)
    print("=" * 10 + "`EventualityExtractor.extract_eventualities_from_parsed_result`" + "=" * 10)
    print("Input: ")
    pprint(parsed_result)
    print("Output: ")
    print(eventualities)

    ##############################################################
    ##
    ## Extract relations from  (parsed results, eventuality list)
    ##
    ##############################################################

    text = "I go to lunch because I am hungry. I go to kitchen before I find some food"
    parsed_results = parse_sentense_with_stanford(text, corenlp_client=e_extractor.corenlp_client)

    sentences = list()
    for parsed_result in parsed_results:
        eventualities = e_extractor.extract_eventualities_from_parsed_result(parsed_result)
        sentences.append(
            (parsed_result, eventualities))

    r_extractor = SeedRuleRelationExtractor()
    pprint(r_extractor.extract(sentences))

    e_extractor.close()
