# ASER: A Large-scale Eventuality Knowledge Graph

ASER (activities, states, events, and their relations), a large-scale eventuality knowledge graph extracted from more than 11-billion-token unstructured textual data. ASER contains 15 relation types belonging to five categories (Temporal, Contingency, Comparison, Expansion, and Co-Occurrence), 194-million unique eventualities, and 64-million unique edges among them.

Project and data homepage is [https://hkust-knowcomp.github.io/ASER](https://hkust-knowcomp.github.io/ASER/).
The latest preprint is on [https://arxiv.org/abs/1905.00270](https://arxiv.org/abs/1905.00270);
You can play the [demo](http://songcpu1.cse.ust.hk/aser/demo).
And the full [documentation](http://songcpu1.cse.ust.hk/aser/document) is comming soon.

### Installation
* Download or clone this repo and then install:
  ```bash
  python setup.py install
  ```
* Download JRE (or JDK) and set your environment variables, such as JAVA_HOME.
* Download Stanford Corenlp 3.9.2 from [CoreNLP](https://stanfordnlp.github.io/CoreNLP/history.html), test it by
  ```bash
  java -mx4g -cp * edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 999999 -port 9000 -quiet false
  ```

### Eventuality and Relation Extraction

```python
from aser.extract.aser_extractor import DiscourseASERExtractor

aser_extractor = DiscourseASERExtractor(corenlp_path="/path/to/corenlp", corenlp_port=9000)
aser_extractor.extract_eventualities_from_text(...)
aser_extractor.extract_relations_from_text(...)
aser_extractor.extract_from_text(...)
```

### Knowledge Graph Database Connection
```python
from aser.database.kg_connection import ASERKGConnection

kg_conn = ASERKGConnection(os.path.join("/path/to/kg_dir", "KG.db"), mode="memory")

kg_conn.get_exact_match_eventuality(...)
kg_conn.get_exact_match_relation(...)
kg_conn.get_related_eventualities(...)

kg_conn.close()
```

### Conceptulization
```python
```

### ASER Client/Server
Besides the offline usage, users can also start an ASER server and connect to ASER from different clients.

1. Start CoreNLP servers as aforementioned.

2. Start an ASER Server
```bash
$ aser-server -n_workers 2 -n_concurrent_back_socks 10 -port 11000 -port_out 11001 \
        -corenlp_path /path/to/corenlp/ \
        -base_corenlp_port 9000 -kg_dir /path/to/kg_dir
```

3. Access ASER from the client
```python
    client = ASERClient(port=11000, port_out=11001)

    s1 = 'I am hungry'
    s2 = 'Evert said'
    # Get Eventualities
    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)

    # Get Relation
    rel = client.predict_relation(event1[0][0], event2[0][0])

    # Get related events
    related_events = client.fetch_related_events(event1[0][0])

    # Get Concept
    concepts = client.conceptualize_event(event1[0][0])
```


### ASER Extraction Pipeline
1. Run the Stanford Corenlp servers on several ports 9000, 9001, ... (assume we use 9000, 9001, 9002, and 9003)
2. Prepare your raw text where two paragraphs (or documents) are separated by two "\n\n".
3. Extract eventualities and relations (four process on ports [9000, 9003] and eight threads):
  ```bash
  aser-pipe -n_workers 8 -corenlp_path "/path/to/corenlp" -base_corenlp_port 9000 -n_extractors 4 \
  -raw_dir "/path/to/raw_dir" -processed_dir "/path/to/processed_dir" \
  -core_kg_dir "/path/to/core_kg_dir" -full_kg_dir "/path/to/full_kg_dir" \
  -eventuality_frequency_lower_cnt_threshold 2 -relation_frequency_lower_cnt_threshold 2 \
  -log_path "/path/to/log_file"
  ```

### Citation
```
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
```