# ASER dialogue response generation task (for ASER v1.0)



## Data

### Splits

```
./data
	train.json
	valid.json
	test.json
```



### Data Format

```python
{
  "post": "" # post string
  "response": ""  # response string
  "post_activity": [] # Eventualities in post
  "response_activity": [] # Eventualities in response, always be empty list
  "post_parsed_relations": [] # the results of StanfordNLP dependency parsing from post
  "aser_triples": [] # triples in ASER
  "omcs_triples": [] # triples in OMCS
  "knowlywood_triples": [] # triples in knowlywood 
}
```



## How to run the code

### Step 1. Build the vocabulary 

```bash
$ python scripts/build_vocab.py data/train.json data/vocab.pt
```



### Step 2. Training

```bash
$ python train.py -config configs/config_seq2seq_attn.json
$ python train.py -config configs/config_aser2seq.json
$ python train.py -config configs/config_omcs2seq.json
$ python train.py -config configs/config_knowly2seq.json
```



### Step 3. Inferring

```bash
$ python infer.py cache/exp_seq2seq_attn/model/best_model.pt data/daily/data/test.json cache/exp_seq2seq_attn/results/pred.test.txt;
$ python infer.py cache/exp_aser2seq/model/best_model.pt data/test.json cache/exp_aser2seq/results/pred.test.txt
$ python infer.py cache/exp_omcs2seq/model/best_model.pt data/test.json cache/exp_omcs2seq/results/pred.test.txt
$ python infer.py cache/exp_knowly2seq/model/best_model.pt data/test.json cache/exp_knowly2seq/results/pred.test.txt
```



### Step 4. Evaluation

```bash
$ perl scripts/multi-bleu.perl data/test.response.txt < cache/exp_seq2seq_attn/results/pred.test.txt
$ perl scripts/multi-bleu.perl data/test.response.txt < cache/exp_aser2seq/results/pred.test.txt
$ perl scripts/multi-bleu.perl data/test.response.txt < cache/exp_omcs2seq/results/pred.test.txt
$ perl scripts/multi-bleu.perl data/test.response.txt < cache/exp_knowly2seq/results/pred.test.txt
```

