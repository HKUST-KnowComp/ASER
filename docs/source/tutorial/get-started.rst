Get Started
================================================================================

.. contents:: :local:


Installation
--------------------------------------------------------------------------------
Currently we only support setup from source. So the first step download this repo

.. highlight:: bash
.. code-block:: bash

    $ git clone https://github.com/HKUST-KnowComp/ASER.git

Then install ASER requirements

.. highlight:: bash
.. code-block:: bash

    $ pip install -r requirements.txt

Finally install ASER as a python package

.. highlight:: bash
.. code-block:: bash

    $ python setup.py install

To run ASER extraction, you need to download corenlp 3.9.2 (2018-10-05) from https://stanfordnlp.github.io/CoreNLP/download.html.

.. highlight:: python
.. code-block:: python

    import urllib
    import zipfile
    import shutil

    urllib.request.urlretrieve("http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip", "stanford-corenlp-3.9.2.zip")
    with zipfile.ZipFile("stanford-corenlp-3.9.2.zip", "r") as zip_ref:
        zip_ref.extractall("./")
    shutil.move("stanford-corenlp-full-2018-10-05", "stanford-corenlp-3.9.2")

To run ASER conceptualizatoin, you need to download Probase from https://concept.research.microsoft.com/Home/Download.

.. highlight:: python
.. code-block:: python

    import urllib
    import zipfile
    import shutil

    urllib.request.urlretrieve("https://concept.research.microsoft.com/Home/DownloadData?key=t9RdYhnkv94TFcd8tkVdzF9cEwNFdaFe&h=602979237", "probase.zip")
    with zipfile.ZipFile("probase.zip", "r") as zip_ref:
        zip_ref.extractall("./")
    shutil.move("data-concept/data-concept-instance-relations.txt", "probase.txt")
    shutil.rmtree("data-concept")

Local Pipeline: aser-pipe
--------------------------------------------------------------------------------

Before getting started, We write three reviews from yelp into a file and want to run the pipepline to build KG.db and concept.db.

.. highlight:: python
.. code-block:: python

    import os

    os.mkdir("raw")
    os.mkdir("processed")
    os.mkdir("core")
    os.mkdir("full")
    os.mkdir("concept")

    with open("raw/yelp.txt", "w") as f:
        f.write("I went there based on reviews on yelp. I was not let down! I got the wild mushroom personal pie and added spinach and fresh jalapenos on the ancient grains crust. It was amazing. The crust was perfectly cooked and all the toppings meshed well together. They have many vegan options(daiya cheese). The owner and the other employee there were very nice and friendly. I will definitely be going back next time I am in town.")
        f.write("\n\n")
        f.write("My experience at this kneaders location was great! I wasn't there during a busy time, about 3:45, but they were very attentive. I placed my to go order and it was ready within 10 minutes. I will definitely be back!")
        f.write("\n\n")
        f.write("I came here for breakfast before a conference, I was excited to try some local breakfast places. I think the highlight of my breakfast was the freshly squeezed orange juice: it was very sweet, natural, and definitely the champion of the day. I ordered the local fresh eggs on ciabatta with the house made sausage. I would have given a four if the bread had been toasted, but it wasnt. The sausage had good flavor, but I would have liked a little salt on my eggs. All in all a good breakfast. If I am back in town I would try the pastries, they looked and smelled amazing.")
        f.write("\n\n")

Then, we can start aser-pipe from command line.

.. highlight:: bash
.. code-block:: bash

    $ aser-pipe -n_extractors 1 -n_workers 1 \
        -corenlp_path "stanford-corenlp-3.9.2" -base_corenlp_port 9000 \
        -raw_dir "raw" -processed_dir "processed" \
        -core_kg_dir "core" -full_kg_dir "full" \
        -eventuality_frequency_threshold 0 -relation_weight_threshold 0 \
        -concept_kg_dir "concept" -concept_method probase -probase_path "probase.txt" \
        -eventuality_threshold_to_conceptualize 0 -concept_weight_threshold 0 -concept_topk 5 \
        -log_path "core/aser_pipe.log"

Now, you can check the core, full, and concept directories.

Step-by-step extraction
--------------------------------------------------------------------------------

Let's see how to utilize an ASER extractor and a conceptualizer.
We provide two kinds of ASERExtractors: the SeedRuleASERExtractor corresponding to the WWW"2020and a new DiscourseASERExtractor which is implemented based on a discourse parsing system.

.. highlight:: python
.. code-block:: python

    from pprint import pprint
    from aser.extract.aser_extractor import SeedRuleASERExtractor, DiscourseASERExtractor

    text = "I came here for breakfast before a conference, I was excited to try some local breakfast places. I think the highlight of my breakfast was the freshly squeezed orange juice: it was very sweet, natural, and definitely the champion of the day. I ordered the local fresh eggs on ciabatta with the house made sausage. I would have given a four if the bread had been toasted, but it wasnt. The sausage had good flavor, but I would have liked a little salt on my eggs. All in all a good breakfast. If I am back in town I would try the pastries, they looked and smelled amazing."
    print("Text:")
    print(text)

    aser_extractor = DiscourseASERExtractor(
      corenlp_path="stanford-corenlp-3.9.2", corenlp_port=9000
    )

    print("-" * 80)
    print("In-order:")
    pprint(aser_extractor.extract_from_text(text, in_order=True))

    print("-" * 80)
    print("Out-of-Order:")
    results = aser_extractor.extract_from_text(text, in_order=False)
    pprint(results)

    eventualities, relations = results

If succeed, you can see the following outputs:

.. highlight:: python
.. code-block:: python

    Text:
    I came here for breakfast before a conference, I was excited to try some local breakfast places. I think the highlight of my breakfast was the freshly squeezed orange juice: it was very sweet, natural, and definitely the champion of the day. I ordered the local fresh eggs on ciabatta with the house made sausage. I would have given a four if the bread had been toasted, but it wasnt. The sausage had good flavor, but I would have liked a little salt on my eggs. All in all a good breakfast. If I am back in town I would try the pastries, they looked and smelled amazing.
    --------------------------------------------------------------------------------
    In-order:
    ([[],
      [i think, it be very sweet],
      [i order, the local fresh egg on ciabatta make sausage],
      [i would have give a four, the bread have be toast],
      [the sausage have good flavor, i would have like a little salt on egg],
      [],
      [i be back in town, i would try the pastry, they look, they smell amazing]],
     [[],
      [(010ec054737a144cb77e99954ff032bc5dff472c, 55704c606666f41a73ac5ae0eabe582892aa163c, {'Co_Occurrence': 1.0})],
      [(b875a4b94675e057fa643beb334e071e4ddf3760, 41876cb7188cb3398572af71ff9d98d61f46c20b, {'Co_Occurrence': 1.0})],
      [(766f00c08dcac14353629c12125f05697eb58a2e, 13bb4ed9f70c37253246c2051ef05fe4795f4fee, {'Co_Occurrence': 1.0, 'Condition': 1.0})],
      [(253e8b127b833c3aa7d79e2b91ce030299a646d6, 8dd8fbc06d2810add7b2cfd637a78f90fa2e5e9e, {'Co_Occurrence': 1.0, 'Contrast': 1.0})],
      [],
      [(dac82e8bc75bd0221e86194e6e3cd607a72aba7e, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Co_Occurrence': 1.0}),
       (dac82e8bc75bd0221e86194e6e3cd607a72aba7e, a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, {'Co_Occurrence': 1.0}),
       (dac82e8bc75bd0221e86194e6e3cd607a72aba7e, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
       (2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, {'Co_Occurrence': 1.0}),
       (2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
       (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
       (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, dac82e8bc75bd0221e86194e6e3cd607a72aba7e, {'Condition': 0.25}),
       (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Condition': 0.25}),
       (1a18ae76468276b651c178926b380e4e9d607f5e, dac82e8bc75bd0221e86194e6e3cd607a72aba7e, {'Condition': 0.25}),
       (1a18ae76468276b651c178926b380e4e9d607f5e, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Condition': 0.25})],
      [],
      [],
      [],
      [],
      [],
      []])
    --------------------------------------------------------------------------------
    Out-of-Order:
    ([i think,
      the bread have be toast,
      they smell amazing,
      the sausage have good flavor,
      i would try the pastry,
      the local fresh egg on ciabatta make sausage,
      it be very sweet,
      i would have give a four,
      i would have like a little salt on egg,
      they look,
      i order,
      i be back in town],
     [(2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
      (1a18ae76468276b651c178926b380e4e9d607f5e, dac82e8bc75bd0221e86194e6e3cd607a72aba7e, {'Condition': 0.25}),
      (253e8b127b833c3aa7d79e2b91ce030299a646d6, 8dd8fbc06d2810add7b2cfd637a78f90fa2e5e9e, {'Co_Occurrence': 1.0, 'Contrast': 1.0}),
      (dac82e8bc75bd0221e86194e6e3cd607a72aba7e, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Co_Occurrence': 1.0}),
      (dac82e8bc75bd0221e86194e6e3cd607a72aba7e, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
      (b875a4b94675e057fa643beb334e071e4ddf3760, 41876cb7188cb3398572af71ff9d98d61f46c20b, {'Co_Occurrence': 1.0}),
      (1a18ae76468276b651c178926b380e4e9d607f5e, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Condition': 0.25}),
      (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, 2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, {'Condition': 0.25}),
      (dac82e8bc75bd0221e86194e6e3cd607a72aba7e, a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, {'Co_Occurrence': 1.0}),
      (766f00c08dcac14353629c12125f05697eb58a2e, 13bb4ed9f70c37253246c2051ef05fe4795f4fee, {'Co_Occurrence': 1.0, 'Condition': 1.0}),
      (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, 1a18ae76468276b651c178926b380e4e9d607f5e, {'Co_Occurrence': 1.0}),
      (010ec054737a144cb77e99954ff032bc5dff472c, 55704c606666f41a73ac5ae0eabe582892aa163c, {'Co_Occurrence': 1.0}),
      (a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, dac82e8bc75bd0221e86194e6e3cd607a72aba7e, {'Condition': 0.25}),
      (2dd66bdf5849fe8d4a28d3355f0fc0a50b7f61e2, a8eec375e86e467cf868a03f64ecd1f9d1fe5fee, {'Co_Occurrence': 1.0})])

As shown above, the in-order will keep the sentence order and token order so that it is a nested list. On the contrary, the out-of-order will return a set of eventualities and a set of relations.

Then, we use the conceptualizer based on probase to conceptualize eventualities.

.. highlight:: python
.. code-block:: python

    from aser.conceptualize.aser_conceptualizer import SeedRuleASERConceptualizer, ProbaseASERConceptualizer
    from aser.conceptualize.utils import conceptualize_eventualities, build_concept_relations

    aser_conceptualizer = ProbaseASERConceptualizer(
      probase_path="probase.txt", probase_topk=5
    )

    cid2concept, concept_instance_pairs, cid_to_filter_score = conceptualize_eventualities(
      aser_conceptualizer, eventualities
    )

    print("-" * 80)
    print("concepts:")
    pprint(list(cid2concept.values()))

    print("-" * 80)
    print("concept_instance_pairs:")
    pprint(concept_instance_pairs)

If succeed, you can see the following outputs:

.. highlight:: python
.. code-block:: python

    --------------------------------------------------------------------------------
    concepts:
    [__PERSON__0 think,
     food toast,
     carbohydrate toast,
     item toast,
     starchy-food toast,
     product toast,
     __PERSON__0 smell amazing,
     meat have flavor,
     sausage have ingredient,
     processed-meat have flavor,
     food have flavor,
     sausage have additive,
     sausage have excipients,
     sausage have factor,
     sausage have characteristic,
     meat-product have flavor,
     item have flavor,
     meat have ingredient,
     processed-meat have ingredient,
     food have ingredient,
     meat have additive,
     meat have excipients,
     meat have factor,
     processed-meat have additive,
     meat have characteristic,
     food have additive,
     processed-meat have excipients,
     processed-meat have factor,
     meat-product have ingredient,
     food have excipients,
     item have ingredient,
     food have factor,
     processed-meat have characteristic,
     food have characteristic,
     meat-product have additive,
     item have additive,
     meat-product have excipients,
     meat-product have factor,
     item have excipients,
     item have factor,
     meat-product have characteristic,
     item have characteristic,
     __PERSON__0 try baked-good,
     __PERSON__0 try food,
     __PERSON__0 try item,
     __PERSON__0 try sweet,
     __PERSON__0 try product,
     food make sausage,
     egg make meat,
     egg make processed-meat,
     egg make food,
     animal-product make sausage,
     egg make meat-product,
     ingredient make sausage,
     food make meat,
     egg make item,
     item make sausage,
     protein make sausage,
     food make processed-meat,
     food make food,
     food make meat-product,
     food make item,
     animal-product make meat,
     ingredient make meat,
     animal-product make processed-meat,
     item make meat,
     protein make meat,
     animal-product make food,
     ingredient make processed-meat,
     ingredient make food,
     item make processed-meat,
     protein make processed-meat,
     item make food,
     protein make food,
     animal-product make meat-product,
     animal-product make item,
     ingredient make meat-product,
     ingredient make item,
     item make meat-product,
     protein make meat-product,
     item make item,
     protein make item,
     it be sweet,
     __PERSON__0 give __NUMBER__0,
     __PERSON__0 like inorganic-contaminant,
     __PERSON__0 like ingredient,
     __PERSON__0 like seasoning,
     __PERSON__0 like item,
     __PERSON__0 like substance,
     __PERSON__0 look,
     __PERSON__0 order,
     __PERSON__0 be area,
     __PERSON__0 be information,
     __PERSON__0 be place,
     __PERSON__0 be location,
     __PERSON__0 be feature]
    --------------------------------------------------------------------------------
    concept_instance_pairs:
    [(__PERSON__0 think, i think, 1.0),
     (food toast, the bread have be toast, 0.17291806206742577),
     (carbohydrate toast, the bread have be toast, 0.047555257870060284),
     (item toast, the bread have be toast, 0.041638758651484704),
     (starchy-food toast, the bread have be toast, 0.04085733422638982),
     (product toast, the bread have be toast, 0.031033712882339807),
     (__PERSON__0 smell amazing, they smell amazing, 1.0),
     (meat have flavor, the sausage have good flavor, 0.13801169590643275),
     (sausage have ingredient, the sausage have good flavor, 0.1330749354005168),
     (processed-meat have flavor,
      the sausage have good flavor,
      0.09395711500974659),
     (food have flavor, the sausage have good flavor, 0.08070175438596491),
     (sausage have additive, the sausage have good flavor, 0.06847545219638243),
     (sausage have excipients, the sausage have good flavor, 0.050387596899224806),
     (sausage have factor, the sausage have good flavor, 0.04909560723514212),
     (sausage have characteristic,
      the sausage have good flavor,
      0.040051679586563305),
     (meat-product have flavor, the sausage have good flavor, 0.03391812865497076),
     (item have flavor, the sausage have good flavor, 0.030019493177387915),
     (meat have ingredient, the sausage have good flavor, 0.018365897517264307),
     (processed-meat have ingredient,
      the sausage have good flavor,
      0.012503337010340954),
     (food have ingredient, the sausage have good flavor, 0.010739380751620654),
     (meat have additive, the sausage have good flavor, 0.009450413285582604),
     (meat have excipients, the sausage have good flavor, 0.006954077700711728),
     (meat have factor, the sausage have good flavor, 0.006775768016078094),
     (processed-meat have additive,
      the sausage have good flavor,
      0.006433755937359909),
     (meat have characteristic, the sausage have good flavor, 0.005527600223642655),
     (food have additive, the sausage have good flavor, 0.005526089124620336),
     (processed-meat have excipients,
      the sausage have good flavor,
      0.004734273236925215),
     (processed-meat have factor,
      the sausage have good flavor,
      0.004612881615465595),
     (meat-product have ingredient,
      the sausage have good flavor,
      0.004513652779666651),
     (food have excipients, the sausage have good flavor, 0.004066367469060248),
     (item have ingredient, the sausage have good flavor, 0.0039948421153371515),
     (food have factor, the sausage have good flavor, 0.003962101636520241),
     (processed-meat have characteristic,
      the sausage have good flavor,
      0.003763140265248248),
     (food have characteristic,
      the sausage have good flavor,
      0.0032322408087401967),
     (meat-product have additive,
      the sausage have good flavor,
      0.002322559197304199),
     (item have additive, the sausage have good flavor, 0.0020555983700278548),
     (meat-product have excipients,
      the sausage have good flavor,
      0.0017090529942427124),
     (meat-product have factor,
      the sausage have good flavor,
      0.0016652311225954636),
     (item have excipients, the sausage have good flavor, 0.0015126101213412515),
     (item have factor, the sausage have good flavor, 0.0014738252464350657),
     (meat-product have characteristic,
      the sausage have good flavor,
      0.00135847802106472),
     (item have characteristic,
      the sausage have good flavor,
      0.0012023311220917638),
     (__PERSON__0 try baked-good, i would try the pastry, 0.29160382101558574),
     (__PERSON__0 try food, i would try the pastry, 0.10155857214680744),
     (__PERSON__0 try item, i would try the pastry, 0.026646556058320763),
     (__PERSON__0 try sweet, i would try the pastry, 0.02262443438914027),
     (__PERSON__0 try product, i would try the pastry, 0.016591251885369532),
     (food make sausage,
      the local fresh egg on ciabatta make sausage,
      0.21908471275559882),
     (egg make meat,
      the local fresh egg on ciabatta make sausage,
      0.13801169590643275),
     (egg make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.09395711500974659),
     (egg make food,
      the local fresh egg on ciabatta make sausage,
      0.08070175438596491),
     (animal-product make sausage,
      the local fresh egg on ciabatta make sausage,
      0.04327599264308125),
     (egg make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.03391812865497076),
     (ingredient make sausage,
      the local fresh egg on ciabatta make sausage,
      0.03321432435356486),
     (food make meat,
      the local fresh egg on ciabatta make sausage,
      0.030236252754573874),
     (egg make item,
      the local fresh egg on ciabatta make sausage,
      0.030019493177387915),
     (item make sausage,
      the local fresh egg on ciabatta make sausage,
      0.027480255328356594),
     (protein make sausage,
      the local fresh egg on ciabatta make sausage,
      0.025749215622633343),
     (food make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.020584567553255093),
     (food make food,
      the local fresh egg on ciabatta make sausage,
      0.01768052067852201),
     (food make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.0074309434735817135),
     (food make item,
      the local fresh egg on ciabatta make sausage,
      0.00657681203983669),
     (animal-product make meat,
      the local fresh egg on ciabatta make sausage,
      0.00597259313670595),
     (ingredient make meat,
      the local fresh egg on ciabatta make sausage,
      0.004583965232421818),
     (animal-product make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.004066087417926932),
     (item make meat,
      the local fresh egg on ciabatta make sausage,
      0.0037925966418082785),
     (protein make meat,
      the local fresh egg on ciabatta make sausage,
      0.0035536929163400405),
     (animal-product make food,
      the local fresh egg on ciabatta make sausage,
      0.0034924485290907673),
     (ingredient make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.003120722093258921),
     (ingredient make food,
      the local fresh egg on ciabatta make sausage,
      0.0026804542460771644),
     (item make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.002581965510383602),
     (protein make processed-meat,
      the local fresh egg on ciabatta make sausage,
      0.0024193220136665247),
     (item make food,
      the local fresh egg on ciabatta make sausage,
      0.0022177048159726376),
     (protein make food,
      the local fresh egg on ciabatta make sausage,
      0.0020780068748090068),
     (animal-product make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.0014678406861395978),
     (animal-product make item,
      the local fresh egg on ciabatta make sausage,
      0.001299123365893667),
     (ingredient make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.0011265677266121413),
     (ingredient make item,
      the local fresh egg on ciabatta make sausage,
      0.0009970771833233895),
     (item make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.0009320788356986446),
     (protein make meat-product,
      the local fresh egg on ciabatta make sausage,
      0.0008733652082530607),
     (item make item,
      the local fresh egg on ciabatta make sausage,
      0.0008249433373424787),
     (protein make item,
      the local fresh egg on ciabatta make sausage,
      0.0007729784027067319),
     (it be sweet, it be very sweet, 1.0),
     (__PERSON__0 give __NUMBER__0, i would have give a four, 1.0),
     (__PERSON__0 like inorganic-contaminant,
      i would have like a little salt on egg,
      0.2110783349721403),
     (__PERSON__0 like ingredient,
      i would have like a little salt on egg,
      0.06014421501147165),
     (__PERSON__0 like seasoning,
      i would have like a little salt on egg,
      0.0429367420517863),
     (__PERSON__0 like item,
      i would have like a little salt on egg,
      0.025401507702392658),
     (__PERSON__0 like substance,
      i would have like a little salt on egg,
      0.022943297279580464),
     (__PERSON__0 look, they look, 1.0),
     (__PERSON__0 order, i order, 1.0),
     (__PERSON__0 be area, i be back in town, 0.06364922206506365),
     (__PERSON__0 be information, i be back in town, 0.033946251768033946),
     (__PERSON__0 be place, i be back in town, 0.03253182461103253),
     (__PERSON__0 be location, i be back in town, 0.0297029702970297),
     (__PERSON__0 be feature, i be back in town, 0.026874115983026876)]

From the conceptualization results, we can find each eventuality would result in multiple concepts. You can use these to make your eventuality representation meaningful.

we do not show build_concept_relations because it requires the concept database. If you are interested, you can use build_concept_relations(aser.database.kg_connection.ASERConceptConnection, List[aser.relation.Relations]).

Do not forget to close your database connections to make the databases available for other processes.

.. highlight:: python
.. code-block:: python

    aser_conceptualizer.close()
    aser_extractor.close()

Client/Server Mode
--------------------------------------------------------------------------------

You can start aser server from command line

.. highlight:: bash
.. code-block:: bash

    $ aser-server -n_workers 1 -n_concurrent_back_socks 10 \
        -port 8000 -port_out 8001 \
        -corenlp_path "stanford-corenlp-3.9.2" -base_corenlp_port 9000 \
        -aser_kg_dir "core" -concept_kg_dir "concept" -probase_path "probase.txt"

Please wait patiently until  `"Loading Server Finished in xx s"` shows up in your console

Now you can access ASER by `ASERClient` from your python code

.. highlight:: python
.. code-block:: python

    from aser.client import ASERClient
    client = ASERClient(port=8000, port_out=8001)

And you can extract the eventualities

.. highlight:: python
.. code-block:: python

    text = "I came here for breakfast before a conference, I was excited to try some local breakfast places. I think the highlight of my breakfast was the freshly squeezed orange juice: it was very sweet, natural, and definitely the champion of the day. I ordered the local fresh eggs on ciabatta with the house made sausage. I would have given a four if the bread had been toasted, but it wasnt. The sausage had good flavor, but I would have liked a little salt on my eggs. All in all a good breakfast. If I am back in town I would try the pastries, they looked and smelled amazing."

    print(client.extract_eventualities(text).__repr__())

It will finally give you this output:

.. highlight:: python
.. code-block:: python

    [[],
     [i think, it be very sweet],
     [i order, the local fresh egg on ciabatta make sausage],
     [i would have give a four, the bread have be toast],
     [the sausage have good flavor, i would have like a little salt on egg],
     [],
     [i be back in town, i would try the pastry, they look, they smell amazing]]

Let's try the relation extraction:

.. highlight:: python
.. code-block:: python

    e1 = client.extract_eventualities("The sausage had good flavor.")[0][0]
    e2 = client.extract_eventualities("I would have liked a little salt on my eggs.")[0][0]

    print(e1.__repr__())
    print(e2.__repr__())
    print(client.predict_eventuality_relation(e1, e2)

And the output should look like the following:

.. highlight:: python
.. code-block:: python

    the sausage have good flavor
    i would have like a little salt on egg
    (253e8b127b833c3aa7d79e2b91ce030299a646d6, 8dd8fbc06d2810add7b2cfd637a78f90fa2e5e9e, {'Contrast': 1.0, 'Co_Occurrence': 1.0})

And we can retrieve the KG to see what eventualities are connected to `e1` (the sausage have good flavor):

.. highlight:: python
.. code-block:: python

    print(client.fetch_related_eventualities(e1))

We get the same relation:

.. highlight:: python
.. code-block:: python

    [(i would have like a little salt on egg,
      (253e8b127b833c3aa7d79e2b91ce030299a646d6, 8dd8fbc06d2810add7b2cfd637a78f90fa2e5e9e, {'Contrast': 1.0, 'Co_Occurrence': 1.0}))]

Conceptualization is powerful to aggregate eventuality information and relations, we can also use the server to do conceptualizatoin and retrieval.

.. highlight:: python
.. code-block:: python

    c1 = client.conceptualize_eventuality(e1)[0][0]
    c2 = client.conceptualize_eventuality(e2)[0][0]

    print(c1.__repr__())
    print(c2.__repr__())
    print(client.predict_concept_relation(c1, c2))

And we can find the *contrast* relation between c1 (meat have flavor) and c2 (__PERSON__0 like inorganic-contaminant):

.. highlight:: python
.. code-block:: python

    meat have flavor
    __PERSON__0 like inorganic-contaminant
    (5a49d855f23b29d0a769d638a0944c0d35815ca9, 86e7181b3e449dd70dd9bd0eebcca5b73b432a8c, {'Contrast': 0.02687880595658219, 'Co_Occurrence': 0.02687880595658219})

Similarly, we can retrieve the concept KG to fetch all neighbors of c1:

.. highlight:: python
.. code-block:: python

    print(client.fetch_related_concepts(c1))

And we are surprising that the concept graph is much denser and more meaningful.

.. highlight:: python
.. code-block:: python

    [(__PERSON__0 like additive,
      (5a49d855f23b29d0a769d638a0944c0d35815ca9, 2342e1896c34cac33974473c5b52ac22d7182fe9, {'Contrast': 0.0019171057650086054, 'Co_Occurrence': 0.0019171057650086054})),
     (__PERSON__0 like item,
      (5a49d855f23b29d0a769d638a0944c0d35815ca9, e8809e959614713c0622e23b0ab5dc06e2f2bf46, {'Contrast': 0.0020252501927783217, 'Co_Occurrence': 0.0020252501927783217})),
     (__PERSON__0 like seasoning,
      (5a49d855f23b29d0a769d638a0944c0d35815ca9, 69864d92726be0d4b7f52fd4f32e38ad1f97974e, {'Contrast': 0.002413587001587757, 'Co_Occurrence': 0.002413587001587757})),
     (__PERSON__0 like ingredient,
      (5a49d855f23b29d0a769d638a0944c0d35815ca9, 58004d785a3ee08eac2f51ea4cbc44bba3a1ba22, {'Contrast': 0.003976765548440927, 'Co_Occurrence': 0.003976765548440927})),
     (__PERSON__0 like inorganic-contaminant,
      (5a49d855f23b29d0a769d638a0944c0d35815ca9, 86e7181b3e449dd70dd9bd0eebcca5b73b432a8c, {'Contrast': 0.02687880595658219, 'Co_Occurrence': 0.02687880595658219}))]

