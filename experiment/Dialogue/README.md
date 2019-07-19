# ASER Dialogue  response generation task



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

