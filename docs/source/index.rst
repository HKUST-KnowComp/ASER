ASER (Activities, States, Events, and their Relations)
================================================================================

.. image:: _static/aser-logo.png
.. image:: _static/aser-demo.png


Introduction
--------------------------------------------------------------------------------

ASER is a large-scale weighted eventuality knowledge graph, including actions, states, events, and their relations.

The eventualities (i.e., nodes of ASER) are extracted using selected dependency patterns. The edges are based on discourse relations (e.g., Result) in discourse analysis.
Besides, conceptualized eventualities in a more abstract level and their relations are also conducted to generalize the knowledge.

In total, ASER (full) contains 438 million eventualities and 648 million edges between eventualities; ASER (core) contains 53 million eventualities and 52 million edges between eventualities.
With the help of Probase (now called Microsoft Concept Graph), ASER (concept) contains 15 million conceptualized eventualities and 224 million edges between conceptualied eventualities.


Data Download
--------------------------------------------------------------------------------

* ASER 2.1: using original text tokens as eventualities and checking the completeness via the dependency parser. [`aws <https://data.dgl.ai/dataset/ASER/README.txt>`_] [`onedrive <https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliucr_connect_ust_hk/Erraz2_KGjFHtbP9bh2-HMoBjCKGYX887MMzLX2y7xbs0w?e=314jRS>`_] [`code <https://github.com/HKUST-KnowComp/ASER/tree/dev>`_]

* `ASER 2.0 <https://arxiv.org/abs/2104.02137>`_ [`aws <https://data.dgl.ai/dataset/ASER/README.txt>`_] [`onedrive <https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliucr_connect_ust_hk/EnlOIunfqNRKsCQIBSXe9pQBL0KhLxTNMSNSJ3Mzt0bmhA?e=xm86PF>`_] [`code <https://github.com/HKUST-KnowComp/ASER/tree/release/2.0>`_]

* `ASER 1.0 <https://arxiv.org/abs/1905.00270>`_ [`onedrive <https://hkustconnect-my.sharepoint.com/:f:/g/personal/xliucr_connect_ust_hk/EoNC-hFNEsNLrZvg73i14e8BMAUDR20TmuLY0W-6tFhKEQ?e=BveOrc>`_] [`code <https://github.com/HKUST-KnowComp/ASER/tree/release/1.0>`_]

* We also provide a copy of Probase from MSRA's official website. All licenses are subject to MSRA's original release. [`onedrive <https://hkustconnect-my.sharepoint.com/:f:/g/personal/zwanggy_connect_ust_hk/Eq5-W3acwqpIrP2xX60C3cgBRxq8dZsgzEcuKl_60ZPaMw?e=DSs7Jb>`_]

Related Projects
--------------------------------------------------------------------------------

* `ASER Extraction, Conceptualization, Usage, and APIs <https://github.com/HKUST-KnowComp/ASER>`_

* `Abstract ATOMIC <https://github.com/HKUST-KnowComp/atomic-conceptualization>`_

* `Benchmarking Commonsense Knowledge Base Population <https://github.com/HKUST-KnowComp/CSKB-Population>`_

* `Transfer to OMCS <https://github.com/HKUST-KnowComp/TransOMCS>`_

* `Transfer to ATOMIC <https://github.com/HKUST-KnowComp/DISCOS-commonsense>`_

* `Eventuality Entailment Graph <https://github.com/HKUST-KnowComp/ASER-EEG>`_

* `FolkScope: Intention Knowledge Graph Construction <https://github.com/HKUST-KnowComp/FolkScope>`_

* `ASER AMIE <https://github.com/HKUST-KnowComp/ASER_AMIE>`_

* `Selectional Preference Annotation (SP-10K) <https://github.com/HKUST-KnowComp/SP-10K>`_

* `Probase <https://haixun.github.io/probase.html>`_, now known as `Microsoft Concept Graph <https://haixun.github.io/probase.html>`_


Talks
--------------------------------------------------------------------------------

* 2023 July: KDD-China. Activity (or Process), State, and Event-based Knowledge Graphs. [`pdf <http://home.cse.ust.hk/~yqsong/papers/2023-KDD-China-YangqiuSong-Final.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/2023-KDD-China-YangqiuSong-Final.pptx>`_]

* 2022 July: Amazon Search Science Team. Acquiring and Modeling Abstract Commonsense Knowledge via Conceptualization. [`pdf <http://home.cse.ust.hk/~yqsong/papers/202207-YangqiuSong-Conceptualization-Final.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/202207-YangqiuSong-Conceptualization-Final.pptx>`_]

* 2021 Novermber: CCKS Tutorial. Commonsense Knowledge Acquisition and Reasoning. Presented by Yangqiu Song. [`pdf <http://home.cse.ust.hk/~yqsong/papers/CCKS2021Tutorial-Commonsense-YangqiuSong.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/CCKS2021Tutorial-Commonsense-YangqiuSong.pptx>`_]

* 2021 September: Huawei Workshop on Commonsense. Commonsense Knowledge Base Population. Presented by Yangqiu Song. [`pdf <http://home.cse.ust.hk/~yqsong/papers/CSKBP-YangqiuSong.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/CSKBP-YangqiuSong.pptx>`_]

* 2021 April: Renmin University and THU. An Overview of Commonsense Knowledge Graph Construction and Reasoning at HKUST. Presented by Yangqiu Song. [`pdf <http://home.cse.ust.hk/~yqsong/papers/ASER-2021.04-ReminUniversity.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/ASER-2021.04-ReminUniversity.pptx>`_]

* 2020 September: NLP with Friends. Commonsense Reasoning from the Angle of Eventualities. Presented by Hongming Zhang. [`ppt <https://hkustconnect-my.sharepoint.com/:p:/g/personal/hzhangal_connect_ust_hk/EbZn6BSgoENIhvuiKWxaZTABh3KfB0IqOqRUI83xgahBvA?e=rlKGvJ>`_] [`video <https://www.youtube.com/watch?v=pI-SiaKsR8w>`_]

* 2020 July: Knowledge Works at Fudan and THU. ASER: Building a Commonsense Knowledge Graph by Higher-order Selectional Preference. Presented by Yangqiu Song. [`pdf <http://home.cse.ust.hk/~yqsong/papers/ASER-202007-KnowledgeWorks.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/ASER-202007-KnowledgeWorks.pptx>`_]

* 2019 October: MSRA. Event-centric Commonsense Reasoning with Structured Knowledge. Presented by Hongming Zhang. [`pdf <https://hkustconnect-my.sharepoint.com/:b:/g/personal/hzhangal_connect_ust_hk/ETe0bW0XPTBBs62-xDi3Qi8B0gqdzQMPZa5VFCEtwEky1Q?e=a3yXjiJ>`_]

* 2019 July: HIT Event Reasoning Workhop, BUPT, PKU, Beihang. ASER: A Large scale Eventuality Knowledge Graph. Presented by Yangqiu Song. [`pdf <http://home.cse.ust.hk/~yqsong/papers/ASER-YangqiuSong.pdf>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/ASER-YangqiuSong.pptx>`_]


Publications
--------------------------------------------------------------------------------

* Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, and Yangqiu Song. Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints. arXiv, abs/2305.19068, 2023. [`pdf <https://arxiv.org/abs/2305.19068>`_]

* Changlong Yu, Weiqi Wang, Xin Liu, Jiaxin Bai, Yangqiu Song, Zheng Li, Yifan Gao, Tianyu Cao, and Bing Yin. FolkScope: Intention Knowledge Graph Construction for E-commerce Commonsense Discovery. Findings of ACL. 2023. [`pdf <https://arxiv.org/abs/2211.08316>`_]

* Tianqing Fang*, Quyet V. Do*, Sehyun Choi, Weiqi Wang, and Yangqiu Song. CKBP v2: An Expert-Annotated Evaluation Set for Commonsense Knowledge Base Population. arXiv, abs/2304.10392, 2023. [`pdf <https://arxiv.org/abs/2304.10392>`_]

* Tianqing Fang, Quyet V. Do, Hongming Zhang, Yangqiu Song, Ginny Y. Wong, and Simon See. PseudoReasoner: Leveraging Pseudo Labels for Commonsense Knowledge Base Population. Findings of EMNLP. 2022. [`pdf <https://arxiv.org/abs/2210.07988>`_]

* Zhaowei Wang, Hongming Zhang, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, and Simon See. SubeventWriter: Iterative Sub-event Sequence Generation with Coherence Controller. Conference on Empirical Methods in Natural Language Processing (EMNLP). 2022. [`pdf <https://arxiv.org/abs/2210.06694>`_]

* Mutian He, Tianqing Fang, Weiqi Wang, and Yangqiu Song. Acquiring and Modelling Abstract Commonsense Knowledge via Conceptualization. arXiv, abs/2206.01532, 2022. [`pdf <https://arxiv.org/abs/2206.01532>`_]

* Hongming Zhang\*, Xin Liu\*, Haojie Pan\*, Haowen Ke, Jiefu Ou, Tianqing Fang, and Yangqiu Song. ASER: Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities. Artificial Intelligence, Volume 309, August 2022, 103740. [`pdf <https://arxiv.org/abs/2104.02137>`_]

* Changlong Yu, Hongming Zhang, Yangqiu Song, and Wilfred Ng. CoCoLM: COmplex COmmonsense Enhanced Language Model. Findings of ACL, 2022. [`pdf <https://arxiv.org/abs/2012.15643>`_]

* Tianqing Fang, Weiqi Wang, Sehyun Choi, Shibo Hao, Hongming Zhang, Yangqiu Song, and Bin He. Benchmarking Commonsense Knowledge Base Population with an Effective Evaluation Dataset. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021. [`pdf <https://arxiv.org/abs/2109.07679>`_]

* Tianqing Fang, Hongming Zhang, Weiqi Wang, Yangqiu Song, and Bin He. DISCOS: Bridging the Gap between Discourse Knowledge and Commonsense Knowledge. The Web Conference (WWW), 2021. [`pdf <https://arxiv.org/abs/2101.00154>`_]

* Hongming Zhang, Muhao Chen, Haoyu Wang, Yangqiu Song, and Dan Roth. Analogous Process Structure Induction for Sub-event Sequence Prediction. Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. [`pdf <https://arxiv.org/abs/2010.08525>`_]

* Changlong Yu, Hongming Zhang, Yangqiu Song, Wilfred Ng, and Lifeng Shang . Enriching Large-Scale Eventuality Knowledge Graph with Entailment Relations. Conference on Automated Knowledge Base Construction (AKBC). 2020. [`pdf <https://openreview.net/forum?id=-oXaOxy6up>`_]

* Hongming Zhang, Daniel Khashabi, Yangqiu Song, and Dan Roth. TransOMCS: From Linguistic Graphs to Commonsense Knowledge. International Joint Conference on Artificial Intelligence (IJCAI). 2020. [`pdf <https://arxiv.org/abs/2005.00206>`_]

* Mutian He, Yangqiu Song, Kun Xu, and Yu Dong. On the Role of Conceptualization in Commonsense Knowledge Graph Construction. HKUST Technical Report, March 6th, 2020. [`pdf <https://arxiv.org/abs/2003.03239>`_]

* Hongming Zhang\*, Xin Liu\*, Haojie Pan\*, Yangqiu Song, and Cane Wing-Ki Leung. ASER: A Large-scale Eventuality Knowledge Graph. The Web Conference (WWW), 2020. [`pdf <https://arxiv.org/abs/1905.00270>`_] [`ppt <http://home.cse.ust.hk/~yqsong/papers/ASER-WWW20.pptx>`_]

* Hongming Zhang, Hantian Ding, and Yangqiu Song. SP-10K: A Large-Scale Evaluation Set for Selectional Preference Acquisition. Annual Meeting of the Association for Computational Linguistics (ACL). 2019. [`pdf <https://arxiv.org/abs/1906.02123>`_]


.. Tutorial
.. --------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorial/get-started


.. API Reference
.. --------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/object
   api/database
   api/extractor
   api/conceptualizer
   api/aser-cs


.. About
.. --------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: About
   :glob:

   about/index


API Index
--------------------------------------------------------------------------------

* :ref:`genindex`

