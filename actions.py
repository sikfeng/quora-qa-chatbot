import gensim
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import TfidfModel
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
from gensim.parsing.preprocessing import STOPWORDS

import numpy as np
import re
import string
import requests
import json
import time
import signal
import sys
import pickle
import traceback
import subprocess
import logging

import quora_answer
import get_similar


logger = logging.getLogger('actions')
logger.setLevel(logging.DEBUG)
ch_logger = logging.StreamHandler()
ch_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch_logger.setFormatter(formatter)
logger.addHandler(ch_logger)

API_KEY = ""
try:
  with open("API_KEY", 'r') as API_KEY_FILE:
    API_KEY = API_KEY_FILE.readline().strip()
    logger.info(f"Retrieved API_KEY {API_KEY}")
except Exception as e:
  logger.critical("Could not get API_KEY!")
  sys.exit(0)

NLU_IP = "http://localhost:5005/model/parse"

## conversations = {user_id: {query, similar_idx, curr_displayed_qns, displayed_msg_id, query_buttons, question_answer}}
conversations = {}

try:
  conversations = pickle.load(open("saved_data/conversations.pkl", "rb"))
  logger.info("Loaded conversations from saved_data/conversations.pkl")
except (OSError, IOError) as e:
  pass

# Send POST request to Rasa NLU to get intent
def POST_ask_nlu(text):

  #response = requests.post(NLU_IP, data='{"text":"' + str(text, "utf-8") + '"}')
  post_data='{"text":"' + text + '"}'
  response = requests.post(NLU_IP, data=post_data.encode('utf-8'), headers={"Content-Type": "application/json; charset=UTF-8"})
  logger.debug(f"NLU response: {response.text}")
  response_json = json.loads(response.text)
  if response_json['intent']['confidence'] < 0.5:
    # assume user put rubbish
    return None
  return response_json['intent']['name']

# Send POST request to Telegram Bot API to get new messages
def POST_get_update(offset=0, limit=100, timeout=0, allowed_updates=[]):
  
  try:
    r = requests.post(f"https://api.telegram.org/bot{API_KEY}/getUpdates", 
                      json={'offset': offset, 'limit': limit, 'timeout': timeout, 
                            'allowed_updates': allowed_updates})
    return r.text
  except requests.exceptions.ConnectionError:
    time.sleep(5)
    # cooldown
    return POST_get_update(offset=offset, limit=limit, timeout=timeout, allowed_updates=[])

  
# Send POST request to Telegram Bot API to send a message
def POST_send_message(chat_id, text, parse_mode="Markdown", disable_web_page_preview=False, 
                disable_notification=False, reply_to_message_id=None, reply_markup=None):

  data = {'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode,
          'disable_web_page_preview': disable_web_page_preview, 
          'disable_notification': disable_notification, 
          'reply_to_message_id': reply_to_message_id}
  if reply_markup is not None:
    data['reply_markup'] = reply_markup
  r = requests.post(f"https://api.telegram.org/bot{API_KEY}/sendMessage", 
                    json=data)
  return r.text


# Send POST request to Telegram Bot API to edit a sent message
def POST_edit_message(chat_id, message_id, text, parse_mode="Markdown", 
                      reply_markup=None):

  data = {'chat_id': chat_id, 'message_id': message_id, 'text': text, 'parse_mode': parse_mode}
  if reply_markup is not None:
    data['reply_markup'] = reply_markup
  r = requests.post(f"https://api.telegram.org/bot{API_KEY}/editMessageText", 
		    json=data)
  return r.text


def get_similar_questions(user_query):

  similar_questions = [question[0].item() for question in get_similar.get_similar(user_query)]
  #print(similar_questions)
  displayed_questions = [0, min(5, len(similar_questions)) - 1] # in case there are less than 5 questions returned for some reason
  return similar_questions, displayed_questions


def highlight_keywords(question, question_idx):
  # given question and idx of question, highlights the words with high idf in question
  remove_str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  translator = str.maketrans(remove_str, ' ' * len(remove_str))

  # split_question is just split on whitespace
  split_question = question.translate(translator).split(" ")
  logger.debug(f'split_question = {split_question}')

  # tokenized_question is split on whitespace and stemmed
  tokenized_question = get_similar.preprocess_text(question)
  logger.debug(f'tokenized_question = {tokenized_question}')
  
  # tokens_idf has idf values of each token
  tokens_idf = get_similar.get_tokens_idf(question_idx)
  logger.debug(f'tokens_idf = {tokens_idf}')
  
  # highlight_idx keeps idx of token that should be highlighted
  highlight_idx = set()

  for idx, token in enumerate(tokenized_question):
    if token in STOPWORDS:
      continue
    for token2, token_idf in tokens_idf.items():
      if token_idf < 5:
        continue
      if token == token2:
        highlight_idx.add(idx)
        break

  highlighted_count = 0
  highlighted_question = ""
  for idx, token in enumerate(split_question):
    if idx in highlight_idx:
      highlighted_count += 1
      highlighted_question += f'[{token}]'
    else:
      highlighted_question += token
    if idx != len(split_question) - 1:
      if question[len(highlighted_question) - highlighted_count*2] != " ":
        highlighted_question += question[len(highlighted_question) - highlighted_count*2]
      else:
        highlighted_question += " "
  return highlighted_question


def print_displayed_questions(similar_questions, displayed_questions):

  text = []
  logger.debug(f'similar_questions = {similar_questions}')
  logger.debug(f'displayed_questions = {displayed_questions}')
  if len(similar_questions) == 0:
    text.append({'text': "No questions match your query",
                 'callback_data': f"/end"})
  else:
    for idx in range(displayed_questions[0], displayed_questions[1] + 1):
      question = highlight_keywords(str(get_similar.questions[similar_questions[idx]]), similar_questions[idx])
      text.append({'text': question,
                   'callback_data': f"/get_query_{len(text)}"})

  buttons = []
  if displayed_questions[0] > 0:  
    buttons.append({
      "text": "Prev ◀️",
      "callback_data": "/prev",
    })
  buttons.append({
    "text": "🆗",
    "callback_data": "/end",
  })
  if displayed_questions[1] < len(similar_questions) - 1:  
    buttons.append({
      "text": "▶️ Next",
      "callback_data": "/next",
    })
  return text, buttons


def print_current(user_id, edit=True):

  similar_questions = conversations[user_id]['similar_questions']
  displayed_questions = conversations[user_id]['displayed_questions']
  queries, scroll_buttons = print_displayed_questions(similar_questions, displayed_questions)
  query_buttons = [[query] for query in queries]

  if conversations[user_id]['displayed_msgs_id'] is None or edit == False:
    query_button_response = json.loads(POST_send_message(
      user_id, 
      f"`{'='*9}\n Results\n{'='*9}\n`{conversations[user_id]['question_answer'] if conversations[user_id]['question_answer'] is not None else ''}", 
      reply_markup={'inline_keyboard': query_buttons}
    ))
    if conversations[user_id]['displayed_questions'][1] == -1:
      scroll_bar_response = json.loads(POST_send_message(
        user_id,
        'None displayed',
        reply_markup={'inline_keyboard': [scroll_buttons]}
      ))
    else:
      scroll_bar_response = json.loads(POST_send_message(
        user_id,
        f'[{displayed_questions[0] + 1}-{displayed_questions[1] + 1}]/{len(similar_questions)} displayed',
        reply_markup={'inline_keyboard': [scroll_buttons]}))
    conversations[user_id]['displayed_msgs_id'] = [query_button_response['result']['message_id'], 
                                                   scroll_bar_response['result']['message_id']]
  else:
    query_button_response = json.loads(POST_edit_message(
      user_id, conversations[user_id]['displayed_msgs_id'][0], 
      f"`{'='*9}\n Results\n{'='*9}\n`{conversations[user_id]['question_answer'] if conversations[user_id]['question_answer'] is not None else ''}",
      reply_markup={'inline_keyboard': query_buttons}
    ))
    if conversations[user_id]['displayed_questions'][1] == -1:
      scroll_bar_response = json.loads(POST_edit_message(
        user_id,
        conversations[user_id]['displayed_msgs_id'][1], 
        'None displayed', 
        reply_markup={'inline_keyboard': [scroll_buttons]}
      ))
    else:
      scroll_bar_response = json.loads(POST_edit_message(
        user_id, 
        conversations[user_id]['displayed_msgs_id'][1],
        f'[{displayed_questions[0] + 1}-{displayed_questions[1] + 1}]/{len(similar_questions)} displayed', 
        reply_markup={'inline_keyboard': [scroll_buttons]}
      ))
  
  conversations[user_id]['displayed_msgs_id'] = [query_button_response['result']['message_id'], 
                                                 scroll_bar_response['result']['message_id']]
  conversations[user_id]['query_buttons'] = query_buttons


def ask_question(user_id, user_query):
  logger.debug(f'{user_id} asked: "{user_query}"')
  similar_questions, displayed_questions = get_similar_questions(user_query)
  conversations[user_id] = {'query': user_query, 'displayed_questions': displayed_questions, 
                            'similar_questions': similar_questions, 'displayed_msgs_id': None,
                            'query_buttons': None, 'question_answer': None}
  print_current(user_id)


def get_next_questions(user_id, edit=True):

  if conversations[user_id]['displayed_questions'][1] < len(conversations[user_id]['similar_questions']) - 1:
    conversations[user_id]['displayed_questions'] = [conversations[user_id]['displayed_questions'][1] + 1,
                                                     min(conversations[user_id]['displayed_questions'][1] + 5,
                                                     len(conversations[user_id]['similar_questions']) - 1)]
    print_current(user_id, edit=edit)


def get_prev_questions(user_id, edit=True):

  if conversations[user_id]['displayed_questions'][0] > 0:
    conversations[user_id]['displayed_questions'] = [max(conversations[user_id]['displayed_questions'][0] - 5, 0), 
                                                     conversations[user_id]['displayed_questions'][0] - 1]
    print_current(user_id, edit=edit)


def end_search(user_id):

  if conversations[user_id]['displayed_msgs_id'] is not None:
    POST_edit_message(user_id, conversations[user_id]['displayed_msgs_id'][1], 
                      f"`{'='*14}\n Search ended\n{'='*14}`")
  conversations[user_id] = {'query': None, 'displayed_questions': None, 
                            'similar_questions': None, 'displayed_msgs_id': None, 'question_answer': None}


def scrape_quora_answer(user_id, user_query):
  # tries to get most similar thing
  if conversations[user_id]['displayed_questions'] is not None:
    selected_question = get_similar.questions[conversations[user_id]['similar_questions'][int(user_query[11:]) +
                                                              conversations[user_id]['displayed_questions'][0]]]
    question_answer, answer_url = quora_answer.get_answer(selected_question)
    logger.debug(f'Answer: {question_answer}')

    # Telegram has max msg length of 4096
    if len(question_answer) > 2000:
      question_answer = question_answer[:2000] + "..."
    question_answer += "\n" + answer_url
    conversations[user_id]['question_answer'] = question_answer
    POST_edit_message(user_id, conversations[user_id]['displayed_msgs_id'][0],
                      text=f"`{'='*9}\n Results\n{'='*9}`\n\n{question_answer}", reply_markup={'inline_keyboard': conversations[user_id]['query_buttons']}) 


def reset_conversation(user_id):
  # resets conversation for user
  logger.debug(f'Reset conversation for user {user_id}')
  if user_id not in conversations:
    conversations[user_id] = {'query': None, 'displayed_questions': None, 
                              'similar_questions': None, 'displayed_msgs_id': None,
                              'query_buttons': None, 'question_answer': None}
    return  
  if 'displayed_msgs_id' in conversations[user_id]:
    if conversations[user_id]['displayed_msgs_id'] is not None:
      POST_edit_message(user_id, conversations[user_id]['displayed_msgs_id'][1], 
                        f"`{'='*14}\n Search ended\n{'='*14}`")
  conversations[user_id] = {
    'query': None, 'displayed_questions': None, 
    'similar_questions': None, 'displayed_msgs_id': None,
    'query_buttons': None, 'question_answer': None
  }


# Determine what function to run according to user response
def parse_response(result):

  if 'callback_query' in result:
    user_id = result['callback_query']['from']['id']
    user_query = result['callback_query']['data']
    if user_id not in conversations:
      reset_conversation(user_id)
    if 'displayed_msgs_id' not in conversations[user_id]:
      # probably leftover responses before reset or user spam clicked
      return
    if conversations[user_id]['displayed_msgs_id'] is None:
      return
    if user_query == '/next':
      get_next_questions(user_id)
      return
    if user_query == '/prev':
      get_prev_questions(user_id)
      return
    if user_query == '/end':
      end_search(user_id)
      return
    if user_query[:11] == '/get_query_':
      scrape_quora_answer(user_id, user_query)
      return

  if 'message' in result:
    user_id = result['message']['from']['id']
    user_query = result['message']['text']
  
    if user_id not in conversations:
      reset_conversation(user_id)

    if user_query[:6] == '/start':
      reset_conversation(user_id)
      POST_send_message(chat_id=user_id, text="Hello :D Dend a question to the bot to start searching")
      return
    #print(result)
    if user_query == '/reset':
      reset_conversation(user_id)
      return
    if user_query[:7] == '/reset ':
      reset_conversation(user_id)
      return
    if user_query == '/end':
      end_search(user_id)
      return
    if user_query[:5] == '/end ':
      end_search(user_id)
      return
    if user_query[:14] == '/ask_question ':
      ask_question(user_id, user_query[14:])
      return
    if user_query == '/prev':
      get_prev_questions(user_id, edit=False)
      return
    if user_query[:6] == '/prev ':
      get_prev_questions(user_id, edit=False)
      return
    if user_query == '/next':
      get_next_questions(user_id, edit=False)
      return
    if user_query[:6] == '/next ':
      get_next_questions(user_id, edit=False)
      return

    if conversations[user_id]['displayed_questions'] is None:
      # user hasn't asked anything, assume this is a question
      ask_question(user_id, user_query)
      return

    # use nlu to predict intent if user sent msg instead of using buttons
    action = POST_ask_nlu(user_query)
    if action == 'end':
      end_search(user_id)
      return
    if action == 'next':
      get_next_questions(user_id, edit=False)
      return
    if action == 'prev':
      get_prev_questions(user_id, edit=False)
      return
    POST_send_message(chat_id=user_id, text="Sorry I did not understand what you said", reply_to_message_id=result['message']['message_id'])

