# ASER2.0 postprocess
All preprocessing code in this dir
1. turn sqlite files to networkx files: `python postprocess_aser/convert_aser_2_nx.py`
2. `python postprocess_aser/rule_based_postprocessing.py` 
    1. turn pronouns, like I, you, he, she, and so on, to PersonX/PersonY/PersonZ
    2. then filter aser with rules: 
       * top 1000 nodes with largest degrees with some exceptions
       * eventualties composed of stopwords
       * eventualities including here, there, this, that, these, those
       * eventualities starting with be, how, t, what, when, where, which, who, why, do, -lsb-, whatever, and if
       * nodes with low/high in/out-degree
       * nodes with URL or/and number
