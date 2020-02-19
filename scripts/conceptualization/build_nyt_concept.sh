#!/usr/bin/env bash

for Year in {1987..2007}
do
   aser_kg_path=/home/data/corpora/aser/database/test_0.3/nyt_$Year
   aser_concept_path=/home/data/corpora/aser/concept/test_0.3/nyt_$Year
   mkdir $aser_concept_path
   python scripts/conceptualization/build_concept.py $aser_kg_path $aser_concept_path
done