#!bin/python3
# coding: utf-8

""" Turret telegram bot
"""

# Standard imports
import json
import requests

# External imports
import telegram
import telegram.ext

import nlu
turretbot = nlu.NLU()

import teleturret.modules.base as base

online_modules = [base]

link = dict()
for m in online_modules:
    for i in m.link.answer_processor.intents:
        link[i] = m.link.answer_processor

# Constants
with open('keys.json') as keys_file:
    keys = json.load(keys_file)
    TELEGRAM_BOT_KEY = keys["telegram"]["teleturretbot"]

# Currently allowed chat ids
ALLOWED_IDS = [
    "224909287",  # @cosmonautd
]

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

updater = telegram.ext.Updater(token=TELEGRAM_BOT_KEY)
dispatcher = updater.dispatcher

def allowed(update):
    if update.message is not None:
        return  str(update.message.chat_id) in ALLOWED_IDS \
                and ( update.message.chat.type == 'private' \
                    or (update.message.chat.type == 'group' and '@teleturretbot' in update.message.text) )
    elif update.callback_query is not None:
        return  str(update.effective_message.chat.id) in ALLOWED_IDS

def build_message(update, type_):
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
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons: menu.insert(0, header_buttons)
    if footer_buttons: menu.append(footer_buttons)
    return menu

def generate_answer(bot, answer, update):
    for a in answer:
        if a['type'] == 'text':
            update.effective_message.reply_text(text=a['text'])
        if a['type'] == 'image':
            update.effective_message.reply_photo(photo=a['url'])
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
    message = build_message(update, type_)
    message_data = turretbot.compute(message['text'])
    message_data['answer'] = link[message_data['intent']].compute(message, message_data)
    generate_answer(bot, message_data['answer'], update)

def start(bot, update):
    if allowed(update):
        bot.send_message(chat_id=update.message.chat_id, text="I see you!")

def answer_text(bot, update):
    if allowed(update):
        teleturretbot(update, 'text', bot)

start_handler = telegram.ext.CommandHandler('start', start)
dispatcher.add_handler(start_handler)

text_handler = telegram.ext.MessageHandler(telegram.ext.Filters.text, answer_text)
dispatcher.add_handler(text_handler)

updater.start_polling()
