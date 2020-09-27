# ASER: A Large-scale Eventuality Knowledge Graph

ASER (activities, states, events, and their relations), a large-scale eventuality knowledge graph extracted from more than 11-billion-token unstructured textual data. ASER contains 15 relation types belonging to five categories (Temporal, Contingency, Comparison, Expansion, and Co-Occurrence), 194-million unique eventualities, and 64-million unique edges among them.

Project and data homepage is [https://hkust-knowcomp.github.io/ASER](https://hkust-knowcomp.github.io/ASER/).
The latest preprint is on [https://arxiv.org/abs/1905.00270](https://arxiv.org/abs/1905.00270);
You can play the [demo](http://songcpu1.cse.ust.hk/aser/demo).
And the full [documentation](http://songcpu1.cse.ust.hk/aser/document) is comming soon.

Citation:

    @inproceedings{ZhangLPSL20,
      author    = {Hongming Zhang and
                  Xin Liu and
                  Haojie Pan and
                  Yangqiu Song and
                  Cane Wing{-}Ki Leung},
      title     = {{ASER:} {A} Large-scale Eventuality Knowledge Graph},
      booktitle = {WWW},
      pages     = {201--211},
      year      = {2020}
    }


### Installation

* Download or clone this repo and then install:
  ```bash
  python setup.py develop
  ```

* Download JRE (or JDK) and set your environment variables, such as JAVA_HOME.
* Download Stanford Corenlp 3.9.2 from [CoreNLP](https://stanfordnlp.github.io/CoreNLP/history.html), test it by
  ```bash
  java -mx4g -cp * edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 999999 -port 9000 -quiet false
  ```

### Pipeline

* Run the Stanford Corenlp servers on several ports 9000, 9001, ... (assume we use 9000, 9001, 9002, and 9003)
* Prepare your raw text where two paragraphs (or documents) are separated by two "\n\n".
* Extract eventualities and relations (four process on ports [9000, 9003] and eight threads):
  ```bash
  aser-pipe -n_workers 8 -corenlp_path "/path/to/corenlp" -base_corenlp_port 9000 -n_extractors 4 \
  -raw_dir "/path/to/raw_dir" -processed_dir "/path/to/processed_dir" \
  -core_kg_dir "/path/to/core_kg_dir" -full_kg_dir "/path/to/full_kg_dir" \
  -eventuality_frequency_lower_cnt_threshold 2 -relation_frequency_lower_cnt_threshold 2 \
  -log_path "/path/to/log_file"
  ```

### Eventuality and Relation Extraction

```python
import os
from aser.extract.aser_extractor import SeedRuleASERExtractor, DiscourseASERExtractor


# SeedRuleASERExtractor corresponds to ASER 1.0,
#   which only uses patterns and rules to extract eventualities and relations
# DiscourseASERExtractor corresponds to ASER 2.0,
#   which uses shallow explicit discourse parser to extract arguments and relations and then further utilizes patterns to extract eventualities
aser_extractor = DiscourseASERExtractor(corenlp_path="/path/to/corenlp", corenlp_port=9000)
aser_extractor.parse_text(...)
aser_extractor.extract_eventualities_from_text(...)
aser_extractor.extract_eventualities_from_parsed_result(...)
aser_extractor.extract_relations_from_text(...)
aser_extractor.extract_relations_from_parsed_result(...)
aser_extractor.extract_from_text(...)
aser_extractor.extract_from_parsed_result(...)

```

### Knowledge Graph Database Connection

```python
import os
from aser.database.kg_connection import ASERKGConnection

# mode can be insert, cache, or memory
# insert is only used to add new elements
# cache stores nothing at the begining but it stores all retrieved eventualities and relations
# memory stores all in memory to speed up querying
kg_conn = ASERKGConnection(os.path.join("/path/to/kg_dir", "KG.db"), mode="memory")

kg_conn.insert_eventualities(...)
kg_conn.insert_relations(...)
kg_conn.get_exact_match_eventuality(...)
kg_conn.get_eventualities_by_keys(...)
kg_conn.get_partial_match_eventualities(...)
kg_conn.get_exact_match_relation(...)
kg_conn.get_relations_by_keys(...)
kgconn.get_related_eventualities(...)

kg_conn.close()
```