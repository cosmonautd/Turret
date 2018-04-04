#!/usr/bin/python3
# coding: utf-8

""" Turret telegram bot
"""

# Standard imports
import json
import logging
import requests

# External imports
import telegram
import telegram.ext

# Project imports
import botkit.nlu

# Teleturret imports
from modules import base

# Initialize botkit, disable entity recognition, activate base module
turretbot = botkit.nlu.NLU(disable=['entities'])
online_modules = [base]

# Get all defined intents and link to their answer processors
link = dict()
for m in online_modules:
    for i in m.link.answer_processor.intents:
        link[i] = m.link.answer_processor

# Load Telegram key and allowed list
with open('config.json') as config_file:
    config = json.load(config_file)
    TELEGRAM_BOT_KEY = config['keys']['telegram']['teleturretbot']
    ALLOWED = list()
    if 'allowed' in config: ALLOWED = config['allowed']

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up Telegram updated and dispatcher
updater = telegram.ext.Updater(token=TELEGRAM_BOT_KEY)
dispatcher = updater.dispatcher

def allowed(update):
    """
    Return True if username is in list ALLOWED
    """
    if update.message is not None:
        return '@' + update.effective_message.from_user.username in ALLOWED \
                and ( update.message.chat.type == 'private' \
                    or (update.message.chat.type == 'group' and '@teleturretbot' in update.message.text) )
    elif update.callback_query is not None:
        return '@' + update.effective_message.from_user.username in ALLOWED

def build_message(update, type_):
    """
    Build a message compatible with botkit
    """
    message = dict()
    message['type'] = type_
    message['username'] = '@' + update.effective_message.from_user.username
    message['userid'] = update.effective_message.chat.id
    if message['type'] == 'text':
        message['text'] = update.message.text.replace('@teleturretbot', '')
    elif message['type'] == 'intent':
        message['intent'] = update.callback_query.data
    return message

def build_menu(buttons, n_cols, header_buttons=None, footer_buttons=None):
    """
    Generate inline menu markup for Telegram
    """
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons: menu.insert(0, header_buttons)
    if footer_buttons: menu.append(footer_buttons)
    return menu

def generate_answer(bot, answer, update):
    """
    Interpret botkit answers and produce Telegram answers
    """
    for a in answer:
        if a['type'] == 'text':
            update.effective_message.reply_text(text=a['text'])
        if a['type'] == 'image':
            update.effective_message.reply_photo(photo=open(a['url'], 'rb'))
        if a['type'] == 'lyrics':
            update.effective_message.reply_text(text=a['lyrics'])
        if a['type'] == 'select':
            button_list = [telegram.InlineKeyboardButton(text=option['text'], callback_data='%s %s' % (option['intent'], option['text'])) for option in a['select']]
            reply_markup = telegram.InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
            bot.send_message(chat_id=update.effective_message.chat.id, text="What do you mean by %s?" % (a['select'][0]['term']), reply_markup=reply_markup)
        if a['type'] == 'link':
            button = [telegram.InlineKeyboardButton(text=a['title'], url=a['link'])]
            reply_markup = telegram.InlineKeyboardMarkup(build_menu(button, n_cols=1))
            bot.send_message(chat_id=update.effective_message.chat.id, text=a['text'], reply_markup=reply_markup)

def teleturretbot(update, type_, bot):
    """
    Get client message, forward to botkit
    Get botkit answers, generate respective Telegram actions
    """
    message = build_message(update, type_)
    message_data = turretbot.compute(message['text'])
    message_data['answer'] = link[message_data['intent']].compute(message, message_data)
    generate_answer(bot, message_data['answer'], update)

def start(bot, update):
    """
    Process /start commmand
    """
    if allowed(update):
        bot.send_message(chat_id=update.message.chat_id, text="I see you!")

def answer_text(bot, update):
    """
    Process arbitrary text
    """
    if allowed(update):
        teleturretbot(update, 'text', bot)

# Set up handler for /start command
start_handler = telegram.ext.CommandHandler('start', start)
dispatcher.add_handler(start_handler)

# Set up handler for arbitrary text
text_handler = telegram.ext.MessageHandler(telegram.ext.Filters.text, answer_text)
dispatcher.add_handler(text_handler)

# I see you
print("Turret Bot ready!")
updater.start_polling()
