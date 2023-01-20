import pandas as pd
import numpy as np
from ml_models import nlp
import yaml
import random
import os
import telebot
from telebot.types import InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup
bot = telebot.TeleBot("5896087941:AAFpFb9ZEQ_omGXc7AC_PqsIxT6FZPRMZ2o")

names_df=pd.read_csv("all_names.csv")

nlp=nlp()
intent_df=nlp.raw_data_import("data\intent_detection.csv")
vocab_list=nlp.create_vocab()
intent_vector=nlp.create_tfidf("data\intent_detection.csv")
intent_tok=nlp.tokenize_intents(intent_df["intent"])
intent_y=nlp.output_series(intent_df,intent_tok)
nlp.naive_bayes_fit(intent_vector,intent_y)
intent_tok_reverse=nlp.invert_dict(intent_tok)

jd=nlp.read_yaml("jd.yaml")
basic_questions=nlp.read_yaml("basic_questions.yaml")
technical_questions=nlp.read_yaml("technical_questions.yaml")
technical_answers=nlp.read_yaml("technical_answers.yaml")
last_intent=None
slot={"name":None,"email": None, "pan":None, "location":None, "exp":None, "edu":None}



final_basic_questions=[]
for i in basic_questions:
    final_basic_questions.append(random.choice(basic_questions[i]))

asked_questions=[]

roles=[i for i in jd["role"]]
roles_sent=",".join(roles)

@bot.message_handler(commands=['hi'])
def send_welcome(message):
    return bot.reply_to(message,"commadn ej ")

        
global greet, ask_interview, name_, email, pan, location, exp, edu,tech_1,tech_2,tech_3,tech_4,eligible_roles,chat_id
greet = True
ask_interview=False
name_=False
email=False
pan=False
location=False
exp=False
edu=False
tech= False
all_tech_= False
tech_3= False
tech_4= False
shuffled_list=[]
eligible_roles=[]
wrong_answer = int(0)



@bot.message_handler(func=lambda msg:greet)
def introduction(message,nlp=nlp):
    user_reply = str(message).lower()
    reply_intent = intent_tok_reverse[nlp.naive_bayes_predict(nlp.vectorize(user_reply))]
    if reply_intent == "greet":
        bot.reply_to(message, f'Hey there! I am Akshay from Innova Solutions. We are currently hiring .Would you like to take an interview for a role?')

        global greet, ask_interview, chat_id
        greet=False
        ask_interview=True
        chat_id=message.chat.id

@bot.message_handler(func=lambda msg:ask_interview)
def introductask_interview(message,nlp=nlp):
    user_reply = str(message).lower()
    reply_intent = intent_tok_reverse[nlp.naive_bayes_predict(nlp.vectorize(user_reply))]
    print(reply_intent)
    if reply_intent == "agree":
        global ask_interview, name_
        ask_interview=False
        name_=True
        if len(final_basic_questions)>0:
            asked_questions.append(final_basic_questions[0])
            return bot.reply_to(message,final_basic_questions.pop(0))
    if reply_intent== "disagree":
        return bot.reply_to(message, "Thank you, see you next time")
    else :
        return bot.reply_to(message, "I am portraying myself as a person, but actually I am a bot, can you please tell me clearly if you want to give the interview ? You will surely save you time if you clearly tell me !!!!")

@bot.message_handler(func=lambda msg:name_)
def name_identification(message,slot=slot,nlp=nlp,last_intent=None):
    user_reply = str(message.text).lower()
    print(user_reply)
    slot["name"]=nlp.identify_name(user_reply, names_df)
    print(slot["name"])
    if slot["name"]!=None:
        print(slot["name"])
        global name_, email
        name_=False
        email=True
        if len(final_basic_questions)>0:
            asked_questions.append(final_basic_questions[0])
            return bot.reply_to(message,final_basic_questions.pop(0))
    if slot["name"]==None:
            return bot.reply_to(message,f'Sorry, i think there has been a misunderstanding my question was "{asked_questions[-1]}"')

@bot.message_handler(func=lambda msg:email)
def email_identification(message,slot=slot,nlp=nlp,last_intent=None):
    user_reply = str(message.text).lower()
    print(user_reply)
    slot["email"]=nlp.check_mail(user_reply)
    print(slot["email"])
    if slot["email"]!=None:
        print(slot["email"])
        global pan, email
        email=False
        pan=True
        if len(final_basic_questions)>0:
            asked_questions.append(final_basic_questions[0])
            return bot.reply_to(message,final_basic_questions.pop(0))
    if slot["email"]==None:
            return bot.reply_to(message,f'Sorry, i think there has been a misunderstanding my question was "{asked_questions[-1]}"')

@bot.message_handler(func=lambda msg:pan)
def pan_identification(message,slot=slot,nlp=nlp,last_intent=None):
    user_reply = str(message.text).lower()
    print(user_reply,"130")
    slot["pan"]=nlp.identify_pan(user_reply)
    print(slot["pan"],"after identification")
    if slot["pan"]!=None:
        print(slot["pan"],"if slot")
        global pan, location
        pan=False
        location=True
        if len(final_basic_questions)>0:
            asked_questions.append(final_basic_questions[0])
            return bot.reply_to(message,final_basic_questions.pop(0))
    if slot["pan"]==None:
            return bot.reply_to(message,f'Sorry, i think there has been a misunderstanding my question was "{asked_questions[-1]}"')

@bot.message_handler(func=lambda msg:location)
def location_identification(message,slot=slot,nlp=nlp,last_intent=None):
    user_reply = str(message.text).lower()
    print(user_reply)
    slot["location"]=(user_reply)
    print(slot["location"])
    if slot["location"]!=None:
        print(slot["location"])
        global exp, location
        exp=True
        location=False
        if len(final_basic_questions)>0:
            asked_questions.append(final_basic_questions[0])
            return bot.reply_to(message,final_basic_questions.pop(0))        
    if slot["location"]==None:
            return bot.reply_to(message,f'Sorry, i think there has been a misunderstanding my question was "{asked_questions[-1]}"')


@bot.message_handler(func=lambda msg:exp)
def exp_identification(message,slot=slot,nlp=nlp,last_intent=None):
    global eligible_roles
    user_reply = str(message.text).lower()
    print(user_reply)
    slot["exp"]=nlp.identify_number(user_reply)
    print(slot["exp"])
    if slot["exp"]!=None:
        print(slot["exp"])
        for i in jd["role"]:
            if int(slot["exp"])>=int(jd["role"][i]["exp"]):
                eligible_roles.append(i)
        global exp, tech
        exp=False
        tech=True
        keyboard = InlineKeyboardMarkup()
        for label in eligible_roles:
            button = InlineKeyboardButton(label, callback_data=label)
            keyboard.add(button)
        return bot.reply_to(message, "I think you will be eligible for one of the follwoing: ", reply_markup=keyboard )

    if slot["exp"]==None:
            return bot.reply_to(message,f'Sorry, i think there has been a misunderstanding my question was "{asked_questions[-1]}"')


@bot.callback_query_handler(func=lambda call: tech)
def handle_callback(call):
    global shuffled_list
    print("call_back")
    shuffled_list = random.sample(technical_questions[call.data], 4)
    if len(shuffled_list)>0:
        question=shuffled_list.pop(0)
        asked_questions.append(question)
        bot.send_message(chat_id, question )
        global tech, all_tech_
        tech=False
        all_tech_=True

@bot.message_handler(func=lambda msg:all_tech_)
def all_tech(message,slot=slot,nlp=nlp,last_intent=None):
        if len(shuffled_list)>0:
            question=shuffled_list.pop(0)
            asked_questions.append(question)
            bot.send_message(message.chat.id, question )
        else:
            bot.send_message(message.chat.id, "it was great talking to you, I will share your details to the hr and get back to you once the interview is evaluated" )
            global all_tech_, greet
            all_tech_=False
            greet=True
bot.infinity_polling()

