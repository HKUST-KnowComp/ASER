import os
from flask import Flask, render_template, request, send_from_directory
from aser.eventuality import Eventuality
from aser.database.kg_connection import relation_senses
from aser.client import ASERClient


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    aser_client = ASERClient(
        ip=os.environ["ASER_HOST"],
        port=int(os.environ["ASER_PORT"]),
        port_out=int(os.environ["ASER_PORT_OUT"]))

    @app.route('/js/<path:path>')
    def send_js(path):
        return send_from_directory('js', path)

    @app.route('/css/<path:path>')
    def send_css(path):
        return send_from_directory('css', path)

    @app.route('/index')
    def index():
        candidates = [
            ("I am patient", request.base_url.replace("index", "search") + "?query=I am patient"),
            ("I went to restaurant before", request.base_url.replace("index", "search") + "?query=I went to restaurant before"),
            ("I have a headache", request.base_url.replace("index", "search") + "?query=I have a headache"),
            ("I am vegan", request.base_url.replace("index", "search") + "?query=I am vegan"),
            ("I have dog", request.base_url.replace("index", "search") + "?query=I have dog"),
            ("You have a car", request.base_url.replace("index", "search") + "?query=You have a car"),
        ]
        return render_template("index.html", events=candidates)

    @app.route('/search', methods=['get'])
    def search():
        if "query" in request.values:
            return get_event(request.values["query"])
        elif "event" in request.values:
            return get_event_with_out_extract(request.values["event"])
        elif "concept" in request.values:
            return get_concept(request.values["concept"])
        else:
            raise RuntimeError

    @app.route('/<sentence>')
    def get_event(sentence):
        data = dict()
        try:
            event = aser_client.extract_eventualities(sentence)[0][0]
        except:
            return render_template("404.html")
        concepts = aser_client.conceptualize_event(event) if event.eid \
                     not in cached_eid_to_concepts else cached_eid_to_concepts[event.eid]
        data["concepts"] = [(" ".join(concept.words) if hasattr(concept, "words") else concept, "{:.3f}".format(score))
                            if concept not in cached_concepts else
                            (" ".join(concept.words) if hasattr(concept, "words") else concept, "{:.3f}".format(score),
                             request.base_url + "?concept=" + "+".join(
                                 concept.words if hasattr(concept, "words") else concept.split(" ")))
                            for concept, score in concepts[:5]]

        data["cached_concepts"] = False
        related_events = aser_client.fetch_related_events(event)
        data["sentence"] = sentence
        data["verbs"] = " ".join(event.verbs)
        data["skeleton_words"] = " ".join(event.skeleton_words)
        data["words"] = " ".join(event.words)
        freq = int(event.frequency)
        data["frequency"] = f'{freq:,}'
        data["show_in_kg"] = "no" if event.frequency <= 0.0 else "yes"
        data["related_events"] = {}
        for rel_e, rels in related_events:
            for rel in rels.relations:
                if rel not in data["related_events"]:
                    data["related_events"][rel] = []
                if len(data["related_events"][rel]) < 20:
                    data["related_events"][rel].append(
                        (" ".join(rel_e.words),
                         request.base_url + "?event=" + rel_e.encode().decode("utf-8")))
        data["rel_keys"] = relation_senses
        return render_template("event.html", event=data)

    @app.route('/<sentence>')
    def get_event_with_out_extract(sentence):
        data = dict()
        event = Eventuality().decode(sentence.encode("utf-8"))
        concepts = aser_client.conceptualize_event(event)
        related_events = aser_client.fetch_related_events(event)
        data["concepts"] = [(" ".join(concept.words), "{:.3f}".format(score))
                            for concept, score in concepts[:5]]
        data["verbs"] = " ".join(event.verbs)
        data["skeleton_words"] = " ".join(event.skeleton_words)
        data["words"] = " ".join(event.words)
        data["sentence"] = data["words"]
        freq = int(event.frequency)
        data["frequency"] = f'{freq:,}'
        data["show_in_kg"] = "no" if event.frequency <= 0.0 else "yes"
        data["related_events"] = {}
        print()
        for rel_e, rels in related_events:
            for rel in rels.relations:
                if rel not in data["related_events"]:
                    data["related_events"][rel] = []
                if len(data["related_events"][rel]) < 20:
                    data["related_events"][rel].append(
                        (" ".join(rel_e.words),
                         request.base_url + "?event=" + rel_e.encode().decode("utf-8")))
        data["rel_keys"] = relation_senses
        return render_template("event.html", event=data)

    @app.route('/<sentence>')
    def get_concept(sentence):
        data = dict()
        print(sentence)
        data["concept"] = sentence
        data["related_concepts"] = dict()
        if sentence in cached_concepts:
            data["rel_keys"] = relation_senses
            for rel_e, rels, _ in cached_concepts[sentence]:
                for rel in rels:
                    if rel not in data["related_concepts"]:
                        data["related_concepts"][rel] = []
                    if len(data["related_concepts"][rel]) < 20:
                        data["related_concepts"][rel].append(
                            (rel_e, ) if rel_e not in cached_concepts else
                            (rel_e,
                             request.base_url + "?concept=" + "+".join(rel_e.split(" "))))
        return render_template("concept.html", event=data)

    return app


cached_eid_to_concepts = {'ece935a0860920c8d1decb4b8276936edfbc8394': [('__PERSON__0 be issue', 1.0)],
                          '34fb3bc62502c0479ba24d6444cde757bb85bbf4': [('__PERSON__0 be diet', 1.0)],
                          'f6c0f6f4ebcf8c5ddc97e1d07176627b06d8279a': [('__PERSON__0 have pet', 1.0)],
                          '6ff86246de6ce5b43fa8d481efc846326c8de541': [('__PERSON__0 be celebrity',
                                                                        0.5),
                                                                       ('__PERSON__0 be performer', 0.5)],
                          '2a89c53726395c01890d24740656b5a682106bbd': [('__PERSON__0 have symptom',
                                                                        1.0)],
                          '6e25b7c1af1ed984da1c38c3be0be34a8db12c98': [('__PERSON__0 have vehicle',
                                                                        1.0)]}

cached_concepts = {'__PERSON__0 be issue': [['__PERSON__0 be proud',
                                             ['Reason', 'Condition', 'Contrast', 'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 be mad',
                                             ['Precedence', 'Synchronous', 'Conjunction'],
                                             14890.47],
                                            ['it be issue',
                                             ['Reason',
                                              'Result',
                                              'Condition',
                                              'Contrast',
                                              'Concession',
                                              'Conjunction',
                                              'Alternative'],
                                             14890.47],
                                            ['__PERSON__0 be sympathetic', ['Conjunction'], 14890.47],
                                            ['__PERSON__0 take explanation', ['Result', 'Conjunction'], 14890.47],
                                            ['__PERSON__0 make information',
                                             ['Precedence', 'Synchronous', 'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 make offer', ['Precedence', 'Conjunction'], 14890.47],
                                            ['__PERSON__0 make discussion', ['Precedence', 'Conjunction'], 14890.47],
                                            ['__PERSON__0 gon be rich', ['Conjunction'], 14890.47],
                                            ['__PERSON__0 be helpless', ['Precedence'], 14890.47],
                                            ['__PERSON__0 be conscious', ['Contrast', 'Conjunction'], 14890.47],
                                            ['__PERSON__0 have diffuse-problem', ['Conjunction'], 14890.47],
                                            ['__PERSON__0 have radically-opposite-answer',
                                             ['Result', 'Condition', 'Contrast', 'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 be ready',
                                             ['Synchronous', 'Reason', 'Result', 'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 be busy',
                                             ['Synchronous', 'Reason', 'Result', 'Condition', 'Contrast',
                                              'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 be sure',
                                             ['Synchronous',
                                              'Reason',
                                              'Result',
                                              'Condition',
                                              'Contrast',
                                              'Concession',
                                              'Conjunction'],
                                             14890.47],
                                            ['service be great', ['Result', 'Contrast', 'Conjunction'], 14890.47],
                                            ['expense be great', ['Result', 'Contrast', 'Conjunction'], 14890.47],
                                            ['issue reverse', ['Synchronous'], 14890.47],
                                            ['__PERSON__0 get resource',
                                             ['Precedence', 'Synchronous', 'Conjunction'],
                                             14890.47],
                                            ['__PERSON__0 be fearless', ['Conjunction'], 14890.47],
                                            ['__PERSON__0 receive information', ['Conjunction'], 14890.47],
                                            ['__PERSON__0 study subject', ['Precedence'], 14890.47]],
                   '__PERSON__0 be diet': [['concept be good', ['Contrast'], 1163.42],
                                           ['element be good', ['Contrast'], 1163.42],
                                           ['__PERSON__0 end up have', ['Conjunction'], 1163.42],
                                           ['__PERSON__0 have positive-emotion', ['Result'], 1163.42],
                                           ['__PERSON__0 be dietary-requirement', ['Reason'], 1163.42],
                                           ['__PERSON__0 be dietary-need', ['Reason'], 1163.42],
                                           ['__PERSON__0 be dietary-restriction', ['Reason'], 1163.42],
                                           ['__PERSON__0 be consumer', ['Reason'], 1163.42],
                                           ['__PERSON__0 be allergic', ['Conjunction'], 1163.42],
                                           ['__PERSON__0 be reward', ['Precedence'], 1163.42],
                                           ['__PERSON__0 be benefit', ['Contrast'], 1163.42],
                                           ['it be delicious',
                                            ['Synchronous', 'Result', 'Contrast', 'Conjunction'],
                                            1163.42],
                                           ['it have ingredient', ['Contrast'], 1163.42],
                                           ['__PERSON__0 enjoy it', ['Contrast'], 1163.42],
                                           ['__PERSON__0 take resource', ['Contrast'], 1163.42],
                                           ['__PERSON__0 be full', ['Contrast'], 1163.42],
                                           ['__PERSON__0 eat animal', ['Synchronous', 'Result', 'Condition'], 1163.42],
                                           ['__PERSON__0 eat protein', ['Synchronous', 'Result', 'Condition'], 1163.42],
                                           ['__PERSON__0 eat protein-rich-food',
                                            ['Synchronous', 'Result', 'Condition', 'Contrast', 'Conjunction'],
                                            1163.42]],
                   '__PERSON__0 have pet': [['it affect __PERSON__0', ['Conjunction'], 1454.89],
                                            ['__PERSON__0 see it', ['Contrast'], 1454.89],
                                            ['__PERSON__0 be young', ['Synchronous'], 1454.89],
                                            ['__PERSON__0 love it', ['Result', 'Condition', 'Conjunction'], 1454.89],
                                            ['__PERSON__0 love animal',
                                             ['Contrast', 'Conjunction', 'Alternative'],
                                             1454.89],
                                            ['__PERSON__0 love mammal', ['Conjunction', 'Alternative'], 1454.89],
                                            ['__PERSON__0 be charity', ['Reason', 'Condition'], 1454.89],
                                            ['__PERSON__0 be lucky', ['Condition', 'Conjunction'], 1454.89],
                                            ['__PERSON__0 have health-problem', ['Condition', 'Conjunction'], 1454.89],
                                            ['__PERSON__0 have serious-complication',
                                             ['Condition', 'Conjunction'],
                                             1454.89],
                                            ['__PERSON__0 have animal',
                                             ['Precedence',
                                              'Reason',
                                              'Condition',
                                              'Contrast',
                                              'Conjunction',
                                              'Alternative'],
                                             1454.89],
                                            ['__PERSON__0 have domestic-animal',
                                             ['Precedence',
                                              'Reason',
                                              'Condition',
                                              'Contrast',
                                              'Conjunction',
                                              'Alternative'],
                                             1454.89],
                                            ['vulnerable-group bear', ['Conjunction'], 1454.89],
                                            ['__PERSON__0 bring asset', ['Conjunction'], 1454.89],
                                            ['__PERSON__0 have emergency', ['Alternative'], 1454.89],
                                            ['__PERSON__0 like pet', ['Precedence', 'Conjunction'], 1454.89],
                                            ['__PERSON__0 like domestic-animal', ['Precedence', 'Conjunction'],
                                             1454.89],
                                            ['__PERSON__0 be lonesome', ['Conjunction'], 1454.89],
                                            ['__PERSON__0 be family-member', ['Reason', 'Conjunction'], 1454.89],
                                            ['__PERSON__0 be kid', ['Succession', 'Synchronous', 'Contrast'], 1454.89]],
                   '__PERSON__0 be celebrity': [['__PERSON__0 be film-genre',
                                                 ['Conjunction'],
                                                 1875.03],
                                                ['__PERSON__0 be professional', ['Contrast', 'Conjunction'], 1875.03],
                                                ['__PERSON__0 be entertainer', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 be prominent-figure',
                                                 ['Precedence', 'Reason', 'Contrast', 'Conjunction', 'Alternative'],
                                                 1875.03],
                                                ['__PERSON__0 be brand', ['Conjunction'], 1875.03],
                                                ['broadway-preformances rable', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 be famous', ['Reason', 'Contrast'], 1875.03],
                                                ['__PERSON__0 be gender-specific-term',
                                                 ['Precedence', 'Condition', 'Restatement'],
                                                 1875.03],
                                                ['__PERSON__0 be famous-person',
                                                 ['Precedence', 'Condition', 'Restatement'],
                                                 1875.03],
                                                ['__PERSON__0 depress', ['Synchronous', 'Conjunction'], 1875.03],
                                                ['__PERSON__0 be impressed', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 be public-figure',
                                                 ['Precedence', 'Reason', 'Contrast', 'Conjunction', 'Alternative'],
                                                 1875.03],
                                                ['__PERSON__0 be performer',
                                                 ['Precedence', 'Reason', 'Contrast', 'Conjunction', 'Alternative'],
                                                 1875.03],
                                                ['__PERSON__0 have issue', ['Conjunction', 'Alternative'], 1875.03],
                                                ['__PERSON__0 love __PERSON__1', ['Contrast'], 1875.03],
                                                ['__PERSON__0 memorize', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 be talented', ['Contrast'], 1875.03],
                                                ['__PERSON__0 be mad', ['Contrast'], 1875.03],
                                                ['it be verbal-behavior', ['Conjunction'], 1875.03],
                                                ['it be inappropriate-act', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 have private-operator', ['Contrast', 'Conjunction'],
                                                 1875.03],
                                                ['__PERSON__0 be artist', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 applaud', ['Conjunction'], 1875.03],
                                                ['__PERSON__0 separate', ['Conjunction'], 1875.03]],
                   '__PERSON__0 be performer': [['__PERSON__0 be person',
                                                 ['Precedence',
                                                  'Reason',
                                                  'Condition',
                                                  'Contrast',
                                                  'Conjunction',
                                                  'Alternative'],
                                                 1005.13],
                                                ['__PERSON__0 be group', ['Conjunction'], 1005.13],
                                                ['there be element', ['Conjunction'], 1005.13],
                                                ['__PERSON__0 want person', ['Result', 'Conjunction'], 1005.13],
                                                ['__PERSON__0 want uninteresting-noun', ['Result', 'Conjunction'],
                                                 1005.13],
                                                ['__PERSON__0 want lone-alias', ['Result', 'Conjunction'], 1005.13],
                                                ['this be word', ['Result', 'Contrast'], 1005.13],
                                                ['this be factor', ['Result', 'Conjunction'], 1005.13],
                                                ['this be term', ['Result'], 1005.13],
                                                ['this be element', ['Result'], 1005.13],
                                                ['__PERSON__0 be public-figure',
                                                 ['Precedence', 'Reason', 'Contrast', 'Alternative'],
                                                 1005.13],
                                                ['__PERSON__0 have issue', ['Alternative'], 1005.13],
                                                ['__PERSON__0 have factor',
                                                 ['Precedence', 'Condition', 'Contrast', 'Conjunction', 'Alternative'],
                                                 1005.13],
                                                ['__PERSON__0 have information',
                                                 ['Condition', 'Contrast', 'Conjunction'],
                                                 1005.13],
                                                ['__PERSON__0 have institution', ['Condition', 'Conjunction'], 1005.13],
                                                ['it catch cognitive-function', ['Contrast'], 1005.13],
                                                ['it catch factor', ['Contrast'], 1005.13],
                                                ['it be unwanted-sexual-behavior', ['Conjunction'], 1005.13],
                                                ['it be inappropriate-act', ['Conjunction'], 1005.13],
                                                ['it be topic', ['Conjunction'], 1005.13],
                                                ['it hai', ['Contrast'], 1005.13],
                                                ['__PERSON__0 be professional', ['Conjunction'], 1005.13],
                                                ['__PERSON__0 applaud', ['Conjunction'], 1005.13],
                                                ['__PERSON__0 learn biblical-story',
                                                 ['Precedence', 'Conjunction'],
                                                 1005.13]],
                   '__PERSON__0 have symptom': [['__PERSON__0 have side-effect',
                                                 ['Synchronous',
                                                  'Reason',
                                                  'Result',
                                                  'Condition',
                                                  'Contrast',
                                                  'Conjunction',
                                                  'Alternative'],
                                                 6589.6],
                                                ['__PERSON__0 have physical-symptom',
                                                 ['Result', 'Contrast', 'Conjunction'],
                                                 6589.6],
                                                ['__PERSON__0 have condition',
                                                 ['Precedence',
                                                  'Synchronous',
                                                  'Reason',
                                                  'Result',
                                                  'Condition',
                                                  'Contrast',
                                                  'Concession',
                                                  'Conjunction',
                                                  'Restatement',
                                                  'Alternative'],
                                                 6589.6],
                                                ['__PERSON__0 have flu-like-symptom',
                                                 ['Result', 'Contrast', 'Conjunction', 'Alternative'],
                                                 6589.6],
                                                ['correlation do issue', ['Conjunction'], 6589.6],
                                                ['__PERSON__0 see case', ['Reason', 'Conjunction'], 6589.6],
                                                ['__PERSON__0 be grumpy', ['Conjunction'], 6589.6],
                                                ['__PERSON__0 wake up',
                                                 ['Precedence',
                                                  'Synchronous',
                                                  'Condition',
                                                  'Contrast',
                                                  'Conjunction',
                                                  'ChosenAlternative'],
                                                 6589.6],
                                                ['__PERSON__0 couldn t sleep', ['Contrast'], 6589.6],
                                                ['__PERSON__0 retire', ['Precedence', 'Conjunction'], 6589.6],
                                                ['__PERSON__0 open product',
                                                 ['Precedence', 'Synchronous', 'Contrast'],
                                                 6589.6],
                                                ['__PERSON__0 faint', ['Conjunction'], 6589.6],
                                                ['it be penalty', ['Result'], 6589.6],
                                                ['it be punishment', ['Result'], 6589.6],
                                                ['__PERSON__0 pass out', ['Precedence', 'Conjunction'], 6589.6],
                                                ['__PERSON__0 see personal-information', ['Precedence'], 6589.6]],
                   '__PERSON__0 have vehicle': [['__PERSON__0 be local-government',
                                                 ['Synchronous', 'Conjunction'],
                                                 977.52],
                                                ['__PERSON__0 have __NUMBER__0',
                                                 ['Reason', 'Result', 'Contrast', 'Conjunction'],
                                                 977.52],
                                                ['__PERSON__0 feel adventurous', ['Contrast'], 977.52],
                                                ['__PERSON__0 visit __CITY__0', ['Synchronous'], 977.52],
                                                ['__PERSON__0 want drink', ['Contrast'], 977.52],
                                                ['__PERSON__0 have asset',
                                                 ['Result', 'Condition', 'Contrast', 'Conjunction'],
                                                 977.52],
                                                ['__PERSON__0 be destination', ['Synchronous'], 977.52],
                                                ['__PERSON__0 have device', ['Contrast'], 977.52],
                                                ['__PERSON__0 have equipment',
                                                 ['Result', 'Condition', 'Contrast', 'Conjunction'],
                                                 977.52],
                                                ['__PERSON__0 have large-item', ['Result', 'Condition'], 977.52],
                                                ['__PERSON__0 be young-person', ['Synchronous'], 977.52],
                                                ['__PERSON__0 ride large-item', ['Result', 'Conjunction'], 977.52],
                                                ['__PERSON__0 ride vehicle', ['Result', 'Conjunction'], 977.52],
                                                ['__PERSON__0 have motor-vehicle', ['Contrast'], 977.52],
                                                ['__PERSON__0 have heavy-vehicle', ['Contrast'], 977.52],
                                                ['__PERSON__0 be __COUNTRY__0', ['Synchronous'], 977.52],
                                                ['__PERSON__0 have expense', ['Conjunction'], 977.52],
                                                ['__PERSON__0 have cost', ['Conjunction'], 977.52],
                                                ['there be public-transportation', ['Contrast', 'Conjunction'], 977.52],
                                                ['there be mode-of-transportation', ['Contrast'], 977.52],
                                                ['there be public-transport', ['Contrast'], 977.52]]}
