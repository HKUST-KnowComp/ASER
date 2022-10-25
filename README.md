# ASER (Activities, States, Events, and their Relations)

![logo](docs/source/_static/aser-logo.png)

ASER is a large-scale weighted eventuality knowledge graph, including actions, states, events, and their relations.

The eventualities (i.e., nodes of ASER) are extracted using selected dependency patterns.
The edges are based on discourse relations (e.g., Result) in discourse analysis.

Besides, conceptualized eventualities in a more abstract level and their relations are also conducted to generalize the knowledge.

In total, ASER (full) contains 438 million eventualities and 648 million edges between eventualities;
ASER (core) contains 53 million eventualities and 52 million edges between eventualities.

With the help of [Probase](https://concept.research.microsoft.com/)  (now called Microsoft Concept Graph), ASER (concept) contains 15 million conceptualized eventualities and 224 million edges between conceptualied eventualities.

The homepage of the project and data is [https://hkust-knowcomp.github.io/ASER](https://hkust-knowcomp.github.io/ASER/).

The online [demo](http://songcpu1.cse.ust.hk/aser/demo) is coming soon.

* ASER 2.1 (dev): using original text tokens as eventualities (set *use_lemma=False* when using extractors) and checking the completeness via the dependency parser. [[code branch](https://github.com/HKUST-KnowComp/ASER/tree/dev)]

* ASER 2.0 (AIJ"2022): ASER: Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities. [[pdf](https://arxiv.org/abs/2104.02137)] [[code branch](https://github.com/HKUST-KnowComp/ASER/tree/release/2.0)]

* ASER 1.0 (WWW"2020): ASER: A Large-scale Eventuality Knowledge Graph. [[pdf](https://arxiv.org/abs/1905.00270)] [[code branch](https://github.com/HKUST-KnowComp/ASER/tree/release/1.0)]

### Quick Start

Please refer to the [get_started.ipynb](examples/get_started.ipynb) or [documentation](https://hkust-knowcomp.github.io/ASER/html/tutorial/get-started.html) to become familiar with ASER and its construction pipeline.

### References
```
@article{ZhangLPKOFS22,
  author    = {Hongming Zhang and
               Xin Liu and
               Haojie Pan and
               Haowen Ke and
               Jiefu Ou and
               Tianqing Fang and
               Yangqiu Song},
  title     = {{ASER:} Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities},
  journal   = {Artificial Intelligence},
  volume    = {309},
  pages     = {103740},
  year      = {2022},
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
