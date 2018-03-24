""" Module for general conversation
"""

import teleturret.answer as answer

class Base:
    """
    """
    def __init__(self):
        self.answer_processor = answer.AnswerProcessor('general')
        self.answer_processor.set_callback('greetings', self.greetings)
        self.answer_processor.set_callback('thanks', self.thanks)
        self.answer_processor.set_callback('goodbye', self.goodbye)
        self.answer_processor.set_callback('whoareyou', self.whoareyou)

    def greetings(self, message, message_data, answer):
        """ Postprocess greetings
        """
        return answer

    def thanks(self, message, message_data, answer):
        """ Postprocess thanks
        """
        return answer

    def goodbye(self, message, message_data, answer):
        """ Postprocess goodbye
        """
        return answer

    def whoareyou(self, message, message_data, answer):
        """ Postprocess whoareyou
        """
        return answer

link = Base()
