import pickle
import gensim
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, stem_text
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import TfidfModel
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
import numpy as np
import re
import string
import logging

import time

logger = logging.getLogger('get_similar')
logger.setLevel(logging.DEBUG)
ch_logger = logging.StreamHandler()
ch_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch_logger.setFormatter(formatter)
logger.addHandler(ch_logger)

# precomputed stuff
questions = None
documents = None
dct = None
corpus = None
tfidf_model = None
corpus_tfidf = None
index = None

#nlp = None
#nlp = spacy.load('en')

def load_precomputed():
  global questions
  global documents
  global dct
  global corpus
  global tfidf_model
  global corpus_tfidf
  global index

  if questions is None:
    questions = pickle.load(open('precompute/questions.pkl', 'rb'))
    logger.info("Loaded questions")

  if documents is None:
    documents = pickle.load(open('precompute/documents.pkl', 'rb'))
    logger.info("Loaded tokenized questions")

  if dct is None:
    dct = pickle.load(open('precompute/dct.pkl', 'rb'))
    logger.info("Loaded dictionary")

  if corpus is None:
    corpus = pickle.load(open('precompute/corpus.pkl', 'rb'))
    logger.info("Loaded corpus")

  if tfidf_model is None:
    tfidf_model = pickle.load(open('precompute/tfidf_model.pkl', 'rb'))
    logger.info("Loaded tfidf model")

  if corpus_tfidf is None:
    corpus_tfidf = pickle.load(open('precompute/corpus_tfidf.pkl', 'rb'))
    logger.info("Loaded tfidf corpus")

  if index is None:
    index = Similarity.load("precompute/similarities.pkl")
    logger.info("Loaded similarities")

  logger.info("Loaded precomputed stuff")


def preprocess_text(document):
  #doc = nlp(document)
  #preprocessed = []
  #for token in doc:
  #  lemmatized = token.lemma_.translate(str.maketrans('', '', string.punctuation))
  #  if lemmatized in ["", " "]:
  #    continue
  #  preprocessed.append(lemmatized)
  #return(preprocessed)
  document = document.rstrip()
  return preprocess_string(document, [lambda x : x.lower(), strip_punctuation, strip_multiple_whitespaces, stem_text])
  
  
def group_tokens(query):
  # manual parsing of string because i cant regex this
  # groups words with 'or' between
  tokenized_query = query.split(' ')
  grouped_tokens = []
  matcher = re.compile(r'\$([0-9]+)')
  for idx, token in enumerate(tokenized_query):
    match = re.match(matcher, token)
    if match:
      if idx > 1 and tokenized_query[idx - 1] == 'or' \
                 and re.match(matcher, tokenized_query[idx - 2]):
        grouped_tokens[-1].append(int(match.group(1)))
      else:
        grouped_tokens.append([int(match.group(1))])
  
  return grouped_tokens


def check_question(query, question):
  # remove all punctuation except spaces and double quotes
  query = query.translate(str.maketrans('', '', "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"))
  query = query.lower()
  
  # replace all keywords with $n so easier to process
  search = re.findall(r'\"([\w\s]+)\"', query)
  if len(search) == 0:
    return True
  for idx, word in enumerate(search):
    query = query.replace(f'"{word}"', f'${idx}')
  
  # group tokens with OR together, should include if it contains at least 1 ov every group
  grouped_tokens = group_tokens(query)
  #print(grouped_tokens)
  for tokens in grouped_tokens:
    appear = False
    for token in tokens:
      if search[token] in question.lower():
        appear = True
        break
    if not appear:
      return False
  return True
 

def get_similar(query):
  tokenized_query = preprocess_text(query)

  query_index = dct.doc2bow(tokenized_query)
  query_tfidf = tfidf_model[query_index]
  query_similarity = index[query_tfidf]

  returned_similar = []
  for similar in query_similarity:
    if check_question(query, questions[similar[0]]):
      returned_similar.append(similar)

  return returned_similar


def get_tokens_idf(question_idx):
  if type(question_idx) is int:
    #tokens = []
    tokens = {}
    for idx, val in corpus_tfidf[question_idx]:
      #tokens.append((dct[idx], val))
      tokens[dct[idx]] = val
    #tokens.sort(key=lambda x: x[1], reverse=True)
    return tokens
  #if type(question_idx) is str:
  
  raise ValueError(f"an integer is required (got type {type(question_idx)})")


#if __name__ == '__main__':
  #start = time.time()
  #print(get_similar("How do I exit \"vim\"?"))
  #print(f'{time.time() - start}s has elapsed')
