from functools import partial

def extract_eventualities_from_parsed_paragraph(event_extractor, paragraph):
    return list(map(event_extractor.extract_from_parsed_result, paragraph))

def extract_relations_from_parsed_paragraph(relation_extractor, paragraph, eventualities_para):
    return relation_extractor.extract(list(zip(paragraph, eventualities_para)), output_format="relation", in_order=True)

def filter_data(processed_data, filtered_eids):
    """ Filter out eventualities whose eids are in filtered_eids
    """
    filtered_data = list()
    for processed_sent in processed_data:
        filtered_sent = list()
        for k, v in processed_sent.items():
            if k != "dependencies":
                filtered_sent[k] = v
            else:
                filtered_es = list()
                for e in v:
                    if e["eid"] not in filtered_eids:
                        filtered_es.append(e)
                filtered_sent[k] = filtered_es
        filtered_data.append(filtered_sent)
    return filtered_data