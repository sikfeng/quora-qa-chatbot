import pandas as pd
import sys
import pickle
import gensim
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, stem_text
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import TfidfModel
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
from tqdm import tqdm
import numpy as np
import re
import string
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


import get_similar

def test():
  point = []
  df_train = pd.read_csv("data/train.csv").astype({"id": int, "qid1": int, "qid2": int, "question1":str, "question2":str, "is_duplicate":bool})
  for row in tqdm(df_train.itertuples()):
    if False:
      continue
    qn1_tokenized = get_similar.preprocess_text(row[4])
    qn1_bow = get_similar.dct.doc2bow(qn1_tokenized)
    qn1_tfidf = get_similar.tfidf_model[qn1_bow]

    qn2_tokenized = get_similar.preprocess_text(row[5])
    qn2_bow = get_similar.dct.doc2bow(qn2_tokenized)
    qn2_tfidf = get_similar.tfidf_model[qn2_bow]
    idx1 = 0
    idx2 = 0
    tfidf1 = []
    tfidf2 = []
    while True:
      if idx1 == len(qn1_tfidf):
        break
      if idx2 == len(qn2_tfidf):
        break
      if qn1_tfidf[idx1][0] == qn2_tfidf[idx2][0]:
        tfidf1.append(qn1_tfidf[idx1][1])
        tfidf2.append(qn2_tfidf[idx2][1])
        idx1 += 1
        idx2 += 1
      elif qn1_tfidf[idx1][0] > qn2_tfidf[idx2][0]:
        tfidf1.append(0)
        tfidf2.append(qn2_tfidf[idx2][1]) 
        idx2 += 1
      elif qn1_tfidf[idx1][0] < qn2_tfidf[idx2][0]:
        tfidf1.append(qn1_tfidf[idx1][1]) 
        tfidf2.append(0)
        idx1 += 1
    while idx1 < len(qn1_tfidf):
      tfidf1.append(qn1_tfidf[idx1][1]) 
      tfidf2.append(0)
      idx1 += 1
    while idx2 < len(qn2_tfidf):
      tfidf1.append(0)
      tfidf2.append(qn2_tfidf[idx2][1]) 
      idx2 += 1
    
    cos_sim = np.dot(tfidf1, tfidf2)/(np.linalg.norm(tfidf1) * np.linalg.norm(tfidf2))
    point.append((cos_sim, row[6]))
 
  point = sorted(point)
  return point

def plot(points, dup):
  sss = "Non-Duplicate"
  if dup:
    sss = "Duplicate"
  plt.figure(1, figsize=(25, 10))
  plt.gca().set(title=f'Similarity of {sss} Question Pair')
  plt.subplot(121)
  plt.hist(points, bins=100, facecolor='blue', alpha=0.5, cumulative=False, weights=np.ones(len(points))/len(points))
  '''
  if dup:
    plt.hist(points, bins=100, facecolor='blue', alpha=0.5, cumulative=False, weights=np.ones(len(points))/len(points))
  else:
    plt.hist(points, bins=100, facecolor='red', alpha=0.5, cumulative=False, weights=np.ones(len(points))/len(points))
  '''
  plt.gca().set(ylabel=f'Percentage of {sss} Pairs', xlabel='Similarity')
  plt.gca().set_xlim([0, 1])
  plt.xticks(np.arange(0, 1.1, 0.1))
  if dup:
    plt.gca().set_ylim([0, 0.04])
    plt.yticks(np.arange(0, 0.045, 0.005))
  else:
    plt.gca().set_ylim([0, 0.1])
    plt.yticks(np.arange(0, 0.11, 0.01))
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)
  plt.subplot(122)
  plt.hist(points, bins=100, facecolor='blue', alpha=0.5, cumulative=True, weights=np.ones(len(points))/len(points))
  '''
  if dup:
    plt.hist(points, bins=100, facecolor='blue', alpha=0.5, cumulative=True, weights=np.ones(len(points))/len(points))
  else:
    plt.hist(points, bins=100, facecolor='red', alpha=0.5, cumulative=True, weights=np.ones(len(points))/len(points))
  '''
  plt.gca().set(ylabel=f'Percentage of {sss} Pairs', xlabel='Similarity')
  plt.gca().set_xlim([0, 1])
  plt.gca().set_ylim([0, 1])
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.yticks(np.arange(0, 1.1, 0.1))
  plt.xticks(np.arange(0, 1.1, 0.1))
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)
  if dup:
    plt.savefig('duplicate.png', dpi=200, bbox_inches='tight')
    print("Saved histogram to duplicate.png")
  else:
    plt.savefig('non-duplicate.png', dpi=200, bbox_inches='tight')
    print("Saved histogram to non-duplicate.png")
  plt.clf()
  plt.cla()


if __name__=='__main__':
  matplotlib.rcParams.update({'font.size': 22})
  #get_similar.load_precomputed()
  #points = test()
  #pickle.dump(points, open("points.pkl", "wb"))
  points = pickle.load(open("points.pkl", "rb"))
  points_d = []
  points_nd = []
  for point, dup in points:
    if isinstance(point, int):
      continue
    if point > 1:
      if dup:
        points_d.append(1)
      else:
        points_nd.append(1)
      continue
    if point < 0:
      if dup:
        points_d.append(0)
      else:
        points_nd.append(0)
      continue
    if dup:
      points_d.append(point)
    else:
      points_nd.append(point)
  plot(points_d, True)
  plot(points_nd, False)
  
