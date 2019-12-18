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
import threading
import queue
import logging

import actions
import get_similar
import quora_answer

logger = logging.getLogger('server')
logger.setLevel(logging.DEBUG)
ch_logger = logging.StreamHandler()
ch_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch_logger.setFormatter(formatter)
logger.addHandler(ch_logger)

# queues = {user_id: queue}
queues = {}
# threads = [thread(process_queue(user_id))]
threads = []

try:
  queues = pickle.load(open("saved_data/queues.pkl", "rb"))
  logger.info("Loaded queues from saved_data/queues.pkl")
except (OSError, IOError) as e:
  pass
  #pickle.dump(queues, open("saved_data/queues.pkl", "wb"))

PIPE = subprocess.PIPE
NLU = subprocess.Popen(['rasa', 'run', '--enable-api', '-m', 'models/model.tar.gz'], stdout=PIPE, stderr=PIPE)

def signal_handler(sig, frame):
  #print(f'Recieved {sig}')
  logger.info(f'Recieved Signal {sig}, Stopping...')
  pickle.dump(queues, open("saved_data/queues.pkl", "wb"))
  pickle.dump(actions.conversations, open("saved_data/conversations.pkl", "wb"))
  for thread in threads:
    thread.join()
  #quora_answer.driver.quit()
  NLU.terminate()
  sys.exit(0)

signal.signal(signal.SIGHUP,  signal_handler)
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGQUIT, signal_handler)
signal.signal(signal.SIGILL,  signal_handler)
signal.signal(signal.SIGTRAP, signal_handler)
signal.signal(signal.SIGABRT, signal_handler)
signal.signal(signal.SIGBUS,  signal_handler)
signal.signal(signal.SIGFPE,  signal_handler)
signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)
signal.signal(signal.SIGUSR2, signal_handler)
signal.signal(signal.SIGPIPE, signal_handler)
signal.signal(signal.SIGALRM, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

get_similar.load_precomputed()
logger.info("Server Initialized")

def process_queue(user_id):
  logger.debug(f'Processing user_id {user_id}')
  if user_id not in queues:
    logger.warning("user_id not in queue, skipping")
    return
  if queues[user_id].empty():
    logger.warning("user_id queue is empty, skipping")
    queues.pop(user_id)
    return
  try:
    update = queues[user_id].get(False)
    actions.parse_response(update)
    if queues[user_id].empty():
      logger.debug(f'user_id {user_id} queue is empty, deleting')
      queues.pop(user_id)
    else:
      process_queue(user_id)
  except Exception as e:
    #print("Error:", e)
    tb = traceback.format_exc()
    print(tb)
    logger.error("Failed to process queue")
    logger.error(f'{e}')
    if not queues[user_id].empty():
      process_queue(user_id)
    queues.pop(user_id)


def add_to_queues(update):
  user_id = 0
  try:
    if 'callback_query' in update:
      user_id = update['callback_query']['from']['id']
    elif 'message' in update:
      user_id = update['message']['from']['id']
    else:
      # idk why would end up here tbh
      return
  except Exception as e:
    logger.error("Could not get user_id of message")
    logger.error(f'{e}')

  if user_id not in queues:
    queues[user_id] = queue.Queue()
    queues[user_id].put(update)
    logger.debug(f'Created user_id {user_id} queue')
    thread = threading.Thread(target=process_queue, args=(user_id,))
    threads.append(thread)
    thread.start()
  else:
    queues[user_id].put(update)
    logger.debug(f'Added user_id {user_id} to queue')


def run():
  latest_update_id = 0
  while True:
    results = None
    try:
      updates = actions.POST_get_update(offset=latest_update_id + 1, timeout=120)
      updates.encode('unicode_escape')
      logger.debug(updates)
      updates_json = json.loads(updates)
      results = updates_json['result']
    except Exception as e:
      logger.error("Failed to get updates")
      logger.error(f'{e}')
      continue
    if results is None:
      logger.error("Failed to get updates")
      logger.error(f'{e}')
      continue
    for result in results:
      logger.debug(f'Recieved update {result}')
      try:
        # get only new messages
        update_id = result['update_id']
        if update_id > latest_update_id:
          latest_update_id = update_id
        else:
          continue
      except KeyError as e:
        logger.error("Failed to get update_id of message")
        logger.error(f'{e}')
        #tb = traceback.format_exc()
        #print(tb)

      try:
        add_to_queues(result)
      except Exception as e:
        logger.error('Failed to add message to queue')
        logger.error(f'{e}')
        #tb = traceback.format_exc()
        #print(tb)
        continue

if __name__ == "__main__":
  run()
