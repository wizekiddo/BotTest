import os
from flask import Flask,request
import pandas as pd
import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from telegram import bot

TOKEN = '837698907:AAHhh1crZSVapOPaSXy26zjajF61mo5MGHk'

bot = telebot.TeleBot(TOKEN)

server = Flask(__name__)

def MLALGOL(testfromuser):
    # Read the data
    df = pd.read_csv('CSV files/news.csv')
    # Get shape and head
    df.shape
    df.head()
    # DataFlair - Get the labels
    labels = df.label
    labels.head()
    # DataFlair - Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # DataFlair - Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # DataFlair - Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # DataFlair - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    # DataFlair - Build confusion matrix
    confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    test = [testfromuser]
    X_test = tfidf_vectorizer.transform(test)
    ynew = pac.predict(X_test)
    str1 = ''.join(ynew)
    return str1


# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg = bot.reply_to(message, """\
Hi there, I am Humo fake news detector.
What do you want to verify?
""")


@bot.message_handler(commands=['help'])
def send_background(message):
    msg = bot.reply_to(message, """\
Humo was developed by Hussain and Mohammed for their senior project in University Of Miami, ECE Dept.
Humo is an attempt to reduce fake news from being spread and increase awareness in social media. Here is how Humo works
1- Copy a title or an article that you want Humo to check.
2- Paste the title or article in Humo's chat
3- Wait couple seconds for Humo to run the algorithm and return the results for you
""")


@bot.message_handler(func=lambda message: True)
def process_check_info(message):
    try:
        chat_id = message.chat.id
        news = message.text
        result = MLALGOL(news)
        query = news.replace("'", '')
        query = query.replace("’", '')
        query = query.replace(' " ', '')
        query = query.replace(',', '')
        query = query.replace('.', '')
        query = query.replace(' ', '+')
        msg = bot.reply_to(message,
                           'it seems this info is ' + "**" + result + "**\n" + 'I also did some research for '
                                                                             'you to '
                                                                             'check ' + 'https://www.snopes.com'
                                                                                        '/?s=' + query)
    except Exception as e:
        bot.reply_to(message, 'Sorry, our data shows no result')


@bot.message_handler(func=lambda message: True)
def handle_messages(messages):
    for message in messages:
        # Do something with the message
        bot.send_message(chat_id=message.chat.id,
                         text="Hold on Humo is running for you, he is still new and can barely walk")
        process_check_info(message)



##bot.set_update_listener(handle_messages)
##bot.polling()
# Enable saving next step handlers to file "./.handlers-saves/step.save".
# Delay=2 means that after any change in next step handlers (e.g. calling register_next_step_handler())
# saving will happen after delay 2 seconds.
##bot.enable_save_next_step_handlers(delay=2)

# Load next_step_handlers from save file (default "./.handlers-saves/step.save")
# WARNING It will work only if enable_save_next_step_handlers was called!
##bot.load_next_step_handlers()


import os

from flask import Flask,request
import pandas as pd
import requests
import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from telegram import bot

TOKEN = '837698907:AAHhh1crZSVapOPaSXy26zjajF61mo5MGHk'

bot = telebot.TeleBot(TOKEN)

server = Flask(__name__)

def MLALGOL(testfromuser):
    # Read the data
    df = pd.read_csv('CSV files/news.csv')
    # Get shape and head
    df.shape
    df.head()
    # DataFlair - Get the labels
    labels = df.label
    labels.head()
    # DataFlair - Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    # DataFlair - Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    # DataFlair - Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # DataFlair - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    # DataFlair - Build confusion matrix
    confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    test = [testfromuser]
    X_test = tfidf_vectorizer.transform(test)
    ynew = pac.predict(X_test)
    str1 = ''.join(ynew)
    return str1


# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg = bot.reply_to(message, """\
Hi there, I am Humo fake news detector.
What do you want to verify?
""")


@bot.message_handler(commands=['help'])
def send_background(message):
    msg = bot.reply_to(message, """\
Humo was developed by Hussain and Mohammed for their senior project in University Of Miami, ECE Dept.
Humo is an attempt to reduce fake news from being spread and increase awareness in social media. Here is how Humo works
1- Copy a title or an article that you want Humo to check.
2- Paste the title or article in Humo's chat
3- Wait couple seconds for Humo to run the algorithm and return the results for you
""")


@bot.message_handler(func=lambda message: True)
def process_check_info(message):
    try:
        chat_id = message.chat.id
        news = message.text
        result = MLALGOL(news)
        query = news.replace("'", '')
        query = query.replace("’", '')
        query = query.replace(' " ', '')
        query = query.replace(',', '')
        query = query.replace('.', '')
        query = query.replace(' ', '+')
        msg = bot.reply_to(message,
                           'it seems this info is ' + "**" + result + "**\n" + 'I also did some research for '
                                                                             'you to '
                                                                             'check ' + 'https://www.snopes.com'
                                                                                        '/?s=' + query)
    except Exception as e:
        bot.reply_to(message, 'Sorry, our data shows no result')


@bot.message_handler(func=lambda message: True)
def handle_messages(messages):
    for message in messages:
        # Do something with the message
        bot.send_message(chat_id=message.chat.id,
                         text="Hold on Humo is running for you, he is still new and can barely walk")
        process_check_info(message)



##bot.set_update_listener(handle_messages)
##bot.polling()
# Enable saving next step handlers to file "./.handlers-saves/step.save".
# Delay=2 means that after any change in next step handlers (e.g. calling register_next_step_handler())
# saving will happen after delay 2 seconds.
##bot.enable_save_next_step_handlers(delay=2)

# Load next_step_handlers from save file (default "./.handlers-saves/step.save")
# WARNING It will work only if enable_save_next_step_handlers was called!
##bot.load_next_step_handlers()
bot.remove_webhook()
bot.polling()
