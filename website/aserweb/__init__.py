import os
from flask import Flask, render_template, request, send_from_directory
from aser.database._kg_connection import relation_senses
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
        return render_template("index.html")

    @app.route('/search', methods=['get'])
    def search():
        return get_event(request.values["query"])

    @app.route('/<sentence>')
    def get_event(sentence):
        data = dict()
        try:
            event = aser_client.extract_eventualities(sentence, only_events=True)[0]
        except:
            return render_template("404.html")
        related_events = aser_client.fetch_related_events(event)
        data["sentence"] = sentence
        data["verbs"] = event["verbs"]
        data["skeleton_words"] = ' '.join([t[0] for t in event["skeleton_words"]])
        data["words"] = ' '.join([t[0] for t in event["words"]])
        data["frequency"] = str(event["frequency"])
        data["show_in_kg"] = "no" if event["frequency"] <= 0.0 else "yes"
        data["related_events"] = {}
        for rel, rel_e in related_events.items():
            if rel not in data["related_events"]:
                data["related_events"][rel] = []
            for i in range(min(len(rel_e), 20)):
                data["related_events"][rel].append(rel_e[i]["words"])
        data["rel_keys"] = relation_senses
        return render_template("event.html", event=data)

    return app