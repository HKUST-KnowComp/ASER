# ASER (Activities, States, Events, and their Relations)

ASER is a large-scale weighted eventuality knowledge graph, including actions, states, events, and their relations.

The eventualities (i.e., nodes of ASER) are extracted using selected dependency patterns.
The edges are based on discourse relations (e.g., Result) in discourse analysis.

Besides, conceptualized eventualities in a more abstract level and their relations are also conducted to generalize the knowledge.

In total, ASER (full) contains 438 million eventualities and 648 million edges between eventualities;
ASER (core) contains 53 million eventualities and 52 million edges between eventualities.

With the help of [Probase](https://concept.research.microsoft.com/)  (now called Microsoft Concept Graph), ASER (concept) contains 15 million conceptualized eventualities and 224 million edges between conceptualied eventualities.

The homepage of the project and data is [https://hkust-knowcomp.github.io/ASER](https://hkust-knowcomp.github.io/ASER/).

The online [demo](http://songcpu1.cse.ust.hk/aser/demo) is coming soon.


* ASER 2.0 (arXiv:2104.02137): ASER: Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities. [[pdf](https://arxiv.org/abs/2104.02137)] [[code branch](https://github.com/HKUST-KnowComp/ASER)]

* ASER 1.0 (WWW"2020): ASER: A Large-scale Eventuality Knowledge Graph. [[pdf](https://arxiv.org/abs/1905.00270)] [[code branch](https://github.com/HKUST-KnowComp/ASER/tree/release/1.0)]

### Quick Start

Please refer to the [get_started.ipynb](examples/get_started.ipynb) or [documentation](http://songcpu1.cse.ust.hk/aser/document) to become familiar with ASER and its construction pipeline.

### References
```
@article{ZhangLPKOFS21,
  author    = {Hongming Zhang and
               Xin Liu and
               Haojie Pan and
               Haowen Ke and
               Jiefu Ou and
               Tianqing Fang and
               Yangqiu Song},
  title     = {{ASER:} Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities},
  journal   = {CoRR},
  volume    = {abs/2104.02137},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.02137},
  archivePrefix = {arXiv},
  eprint    = {2104.02137},
}

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
