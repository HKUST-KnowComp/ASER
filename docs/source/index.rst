Welcome to ASER Tutorial and Documentation
================================================================================

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   section/get-started

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :glob:

   api/database
   api/extractor
   api/conceptualizer
   api/aser-server
   api/aser-client

ASER (Action, States, Events, and their Relations): a large-scale weighted eventuality knowledge graph.

The eventualities (i.e., nodes of ASER) are extracted using selected dependency patterns. The edges are based on discourse relations (e.g., Result) in discourse analysis.
Besides, conceptualized eventualities in a more abstract level and their relations are also conducted to generalize the knowledge.

In total, ASER (full) contains 438 million eventualities and 648 million edges between eventualities; ASER (core) contains 53 million eventualities and 52 million edges between eventualities.
With the help of Probase (now called Microsoft Concept Graph), ASER (concept) contains 15 million conceptualized eventualities and 224 million edges between conceptualied eventualities.

The homepage of the project and data is https://hkust-knowcomp.github.io/ASER.

* ASER 2.0 (arXiv:2104.02137): ASER: Towards Large-scale Commonsense Knowledge Acquisition via Higher-order Selectional Preference over Eventualities. [`pdf2 <https://arxiv.org/abs/2104.02137>`_] [`code branch2 <https://github.com/HKUST-KnowComp/ASER>`_]

* ASER 1.0 (WWW"2020): ASER: A Large-scale Eventuality Knowledge Graph. [`pdf1 <https://arxiv.org/abs/1905.00270>`_] [`code branch1 <https://github.com/HKUST-KnowComp/ASER/tree/release/1.0>`_]

.. image:: _static/aser-demo.png




Index
--------------------------------------------------------------------------------

* :ref:`genindex`
