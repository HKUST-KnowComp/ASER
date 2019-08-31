import json
import os

from flask import Flask, render_template, request
from aser.database._kg_connection import relation_senses
from aser.client import ASERClient


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    aser_client = ASERClient(port=8000)

    @app.route('/index')
    def index():
        return render_template("index.html")

    @app.route('/search', methods=['get'])
    def search():
        return get_event(request.values["query"])

    @app.route('/<sentence>')
    def get_event(sentence):
        data = dict()
        event = aser_client.extract_eventualities(sentence)
        if not event:
            return render_template("404.html")
        exact_event = aser_client.get_exact_match_event(event)
        event = exact_event if exact_event else event
        related_events = aser_client.get_related_events(event)
        data["sentence"] = sentence
        data["verbs"] = event["verbs"]
        data["skeleton_words"] = event["skeleton_words"]
        data["words"] = event["words"]
        data["frequency"] = str(event["frequency"])
        data["show_in_kg"] = "no" if exact_event is None else "yes"
        data["related_events"] = {}
        for rel, rel_e in related_events.items():
            if rel not in data["related_events"]:
                data["related_events"][rel] = []
            for i in range(min(len(rel_e), 20)):
                data["related_events"][rel].append(rel_e[i]["words"])
        data["rel_keys"] = relation_senses
        return render_template("event.html", event=data)

    return app