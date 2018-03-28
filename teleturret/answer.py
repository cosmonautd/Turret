import json
import random

class AnswerProcessor:
    """
    """
    def __init__(self, cluster):
        self.cluster = cluster
        self.intents = self.__get_intents__('training.json')
        self.callbacks = self.__init_callbacks__()
        self.templates = self.__get_templates__('templates.json')

    def __get_intents__(self, training_path):
        intents = list()
        with open(training_path) as f:
            training = json.load(f)
        for sample in training['samples']:
            if sample['cluster'] == self.cluster:
                intents.append(sample['intent'])
        return list(set(intents))

    def __init_callbacks__(self):
        callbacks = dict()
        for intent in self.intents:
            callbacks[intent] = None
        return callbacks

    def __get_templates__(self, templates_path):
        templates = dict()
        with open(templates_path) as f:
            templates_all = json.load(f)
        for intent in templates_all.keys():
            if intent in self.intents:
                templates[intent] = templates_all[intent]
        return templates

    def set_callback(self, intent, callback):
        self.callbacks[intent] = callback

    def compute(self, message, message_data):
        answer_text = random.choice(self.templates[message_data['intent']])
        answer_text = answer_text.replace('<NAME>', message['username'])
        answer = [{'type': 'text', 'text': answer_text}]
        return self.callbacks[message_data['intent']](message, message_data, answer)
