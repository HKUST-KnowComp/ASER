Getting Start
=============

.. contents:: :local:


Installation
------------
Currently we only support setup from source. So the first step download this repo

.. highlight:: bash
.. code-block:: bash

    $ git clone https://github.com/yqsong-hkust/ASER-core.git

Then install ASER requirements

.. highlight:: bash
.. code-block:: bash

    $ pip install -r requirements.txt

Finally install ASER as a python package

.. highlight:: bash
.. code-block:: bash

    $ python setup.py install


Start an aser-server
--------------------

You can start aser server from command line

.. highlight:: bash
.. code-block:: bash

    $ aser-server -n_workers 2 -n_concurrent_back_socks 10 -port 11000 -port_out 11001 \
        -corenlp_path /home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/ \
        -base_corenlp_port 9000 -kg_dir /data/hjpan/ASER/tiny

Please wait patiently until  `"Loading Server Finished in xx s"` shows up in your console


.. note:: Currently we have run a server on songcpu4.cse.ust.hk with port 20002 and port_out 20003 for our group.


Access ASER via aser-client
---------------------------
Now you can access ASER from your python code

.. highlight:: python
.. code-block:: python

    from aser.client import ASERClient
    client = ASERClient(ip="songcpu4.cse.ust.hk", port=20002, port_out=20003)

And you can extract the eventualities

.. highlight:: python
.. code-block:: python

    client.extract_eventualities("I am hungry")

It will finally give you this output:

.. highlight:: python
.. code-block:: python

    {
        'sentence': 'I am hungry',
        'eventualities': [
            {
                'eid': 'c08b06c1b3a3e9ada88dd7034618d0969ae2b244',
                'pattern': 's-be-a',
                'verbs': 'be',
                'frequency': 0.0,
                'skeleton_words': 'i be hungry',
                'words': 'i be hungry'
            }
        ]
    }
