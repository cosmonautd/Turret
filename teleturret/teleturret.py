#!.teleturret-env/bin/python3
# coding: utf-8

""" Turret telegram bot
"""

# Standard imports
import os
import sys
import json
import time
import logging
import requests
import datetime
import threading
import multiprocessing

# External imports
import cv2
import numpy
import scipy
import telegram
import telegram.ext
import face_recognition as fc
import matplotlib
import matplotlib.pyplot as plt

# Project imports
import botkit.nlu

# Teleturret imports
from modules import base

def log(m):
    print(m)
    sys.stdout.flush()

# Initialize botkit, disable entity recognition, activate base module
turretbot = botkit.nlu.NLU(disable=['entities'])
online_modules = [base]

# Get all defined intents and link to their answer processors
link = dict()
for m in online_modules:
    for i in m.link.answer_processor.intents:
        link[i] = m.link.answer_processor
        log("Loaded intent: " + i)

def loadconfig():
    """
    """
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config

def allowed(update):
    """
    Return True if username is in list ALLOWED
    """
    config = loadconfig()
    if 'allowed' in config: ALLOWED = config['allowed']
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
    log('----------------------------------------')
    log('*--------------------------------------*')
    log(json.dumps(message_data, indent=4, sort_keys=True, ensure_ascii=False))
    generate_answer(bot, message_data['answer'], update)

def start(bot, update):
    """
    Process /start commmand
    """
    BOT = bot
    if allowed(update):
        bot.send_message(chat_id=update.message.chat_id, text="I see you!")

def answer_text(bot, update):
    """
    Process arbitrary text
    """
    context = botkit.nlu.Context()
    if not context.has_user('@' + update.effective_message.from_user.username):
        context.write('@' + update.effective_message.from_user.username, 'chat_id', update.effective_message.chat.id)
    if allowed(update):
        teleturretbot(update, 'text', bot)

def im2float(im):
    """
    Convert OpenCV image type to numpy.float
    """
    info = numpy.iinfo(im.dtype)
    return im.astype(numpy.float) / info.max

# Set up notifications
def notifications():
    """
    Dispatch notifications
    """
    # Get path to todays' detections
    now = datetime.datetime.now()
    todaypath = '/'.join(('..', 'detected', str(now.year), str(now.month) + '. ' + now.strftime('%B'), str(now.day)))
    # Get paths for all frames detected today
    detections = list()
    for (_, _, filenames) in os.walk(todaypath):
        detections.extend(filenames)
        break
    detections.sort(reverse=True)
    detections = [d for d in detections if not d.endswith('.log')]
    # If a detection was made today
    if len(detections) > 0:
        # If there was a detection today, get the last frame
        lastframepath = os.path.join(todaypath, detections[0])
        lastframe = cv2.imread(lastframepath)
        # Check the state of lights
        light_lvl = numpy.mean(im2float(lastframe).flatten())
        # Infer if there is someone in the lab
        light = True if light_lvl > 0.3 else False
    
        context = botkit.nlu.Context()
        if not context.has_key('@teleturretbot', 'light'):
            log('New light')
            context.write('@teleturretbot', 'light', light)

        if light != context.read('@teleturretbot', 'light'):
            context.write('@teleturretbot', 'light', light)
            nt_message = 'Someone just %s the lab!' % ('opened' if light else 'closed')
            for username in context.__load__().keys():
                if context.read(username, 'notifications'):
                    bot.send_message(chat_id=context.read(username, 'chat_id'), text=nt_message)

nt_timer = None
def notifications_loop():
    global nt_timer
    now = datetime.datetime.now()
    next_nt_time = now + datetime.timedelta(seconds=5)
    notifications()
    nt_timer = threading.Timer((next_nt_time-now).seconds, notifications_loop)
    nt_timer.setDaemon(True)
    nt_timer.start()
notifications_loop()

database = None
facedatabase = None
facedatabase_names = None
facedatabase_encodings = None
fraction = 1.0
# Set up recognizer if not ready
if (not database) or (not facedatabase) or (not facedatabase_encodings):
    database = list()
    for (_, _, filenames) in os.walk('faces'):
        database.extend(filenames)
        break
    database = sorted(database)
    facedatabase = [fc.load_image_file(os.path.join('faces', name)) for name in database]
    facedatabase_names = [name.split('.')[0] for name in database]
    facedatabase_encodings = [fc.face_encodings(face)[0] for face in facedatabase]

def face_recognition(t_datetime):
    """ Face recognition
    """
    nkeyframes = 10
    # Get path to todays' activity log
    Y, M, M_str, D = t_datetime.year, t_datetime.month, t_datetime.strftime('%B'), t_datetime.day
    h, m, s = t_datetime.hour, t_datetime.minute, t_datetime.second
    todaypath = '/'.join(('..', 'detected', str(Y), str(M) + '. ' + M_str, str(D)))
    for ms in range(1000):    
        hms = str(t_datetime)[:10] + ' ' + '%02d'%(h) + 'h' + '%02d'%(m) + 'm' + '%02d.%03d'%(s,ms) + 's' + '.jpg'
        baseframepath = os.path.join(todaypath, hms)
        if os.path.exists(baseframepath):
            break
    # Get paths for all frames detected today
    detections = list()
    for (_, _, filenames) in os.walk(todaypath):
        detections.extend(filenames)
        break
    detections = [d for d in detections if not d.endswith('.log')]
    detections.sort()
    baseindex = detections.index(hms)
    keyframespaths = detections[baseindex-nkeyframes:baseindex+nkeyframes]
    votes = numpy.zeros(len(database))
    unknown_count = 0
    # Loop through key frames
    for keyframepath in keyframespaths:
        framepath = os.path.join(todaypath, keyframepath)
        frame = cv2.imread(framepath)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (0, 0), fx=fraction, fy=fraction)
        face_locations = fc.face_locations(frame)
        if len(face_locations) > 0:
            face_encodings = fc.face_encodings(frame, face_locations)
            for face_encoding in face_encodings:
                results = fc.compare_faces(facedatabase_encodings, face_encoding, tolerance=0.5)
                if True in results:
                    for i, r in enumerate(results):
                        if r: votes[i] += 1
                else: unknown_count += 1
    if numpy.max(votes) > unknown_count:
        detected_name = facedatabase_names[numpy.argmax(votes)]
    else: detected_name = 'Unknown'

    context = botkit.nlu.Context()
    if not context.has_key('@teleturretbot', 'activity'):
        context.write('@teleturretbot', 'activity', {})
    activity = context.read('@teleturretbot', 'activity')
    name = detected_name
    time = '%04d/%02d/%02d %02d:%02d:%02d' % (Y,M,D,h,m,s)
    activity[time] = name
    context.write('@teleturretbot', 'activity', activity)
    time_str = str(t_datetime)[:19]
    log('%s %s' % (time_str, name))

def event_detection():
    time_window = 8
    while True:
        # Get path to todays' activity log
        now = datetime.datetime.now()
        Y, M, M_str, D = now.year, now.month, now.strftime('%B'), now.day
        logpath = '/'.join(('..', 'detected', str(Y), str(M) + '. ' + M_str, str(D), 'activity.log'))
        if os.path.exists(logpath):
            with open(logpath, 'r') as activity_log:
                activity_times = [line.split()[1] for line in activity_log.readlines()]
                counts = numpy.zeros(time_window)
                # Check activity and accumulate counts
                for t in activity_times[::-1]:
                    h = int(t.split(':')[0])
                    m = int(t.split(':')[1])
                    s = int(t.split(':')[2])
                    t_datetime = datetime.datetime(Y, M, D, h, m, s, 0)
                    shift_datetime = now - t_datetime
                    if shift_datetime < datetime.timedelta(seconds=time_window):
                        index = shift_datetime.seconds
                        counts[index] += 1
                # Identify peaks
                peaks, _ = scipy.signal.find_peaks(counts, height=3, distance=3)
                xaxis = numpy.arange(0, len(counts))
                # Generate graph
                fig, ax = plt.subplots()
                ax.plot(xaxis, counts)
                ax.plot(peaks, counts[peaks], "x")
                ax.set(xlabel='Time', ylabel='Detections', title='Activity Graph')
                # formatter = matplotlib.ticker.FuncFormatter(lambda s, x: '%02d:%02d' % (s//3600,(s%3600)//60))
                # ax.xaxis.set_major_formatter(formatter)
                fig.savefig(".activity-tmp.png", dpi=300, bbox_inches='tight')
                plt.close()
                if len(peaks) > 0:
                    for peak in peaks:
                        activity_peak_datetime = now - datetime.timedelta(seconds=int(peak))
                        face_recognition_process = multiprocessing.Process(target=face_recognition, args=(activity_peak_datetime,))
                        face_recognition_process.start()
                        
            time.sleep(7)
        else:
            time.sleep(60)

event_detection_process = multiprocessing.Process(target=event_detection, args=())
event_detection_process.start()

# Load config file
config = loadconfig()

# Set Telegram API key
TELEGRAM_BOT_KEY = config['keys']['telegram']['teleturretbot']
# Set list of allowed users
ALLOWED = list()
if 'allowed' in config: ALLOWED = config['allowed']

# Set up Telegram updated and dispatcher
bot = telegram.Bot(TELEGRAM_BOT_KEY)
updater = telegram.ext.Updater(bot=bot)
dispatcher = updater.dispatcher

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up handler for /start command
start_handler = telegram.ext.CommandHandler('start', start)
dispatcher.add_handler(start_handler)

# Set up handler for arbitrary text
text_handler = telegram.ext.MessageHandler(telegram.ext.Filters.text, answer_text)
dispatcher.add_handler(text_handler)

# I see you
print("Turret Bot ready!")
updater.start_polling()
