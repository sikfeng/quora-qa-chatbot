import gensim
import pickle
from gensim import corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim import similarities
from gensim.corpora.textcorpus import TextCorpus
from gensim.test.utils import datapath, get_tmpfile
from gensim.similarities import MatrixSimilarity, Similarity
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_multiple_whitespaces, stem_text
import subprocess
import get_similar


def main():
  orig_qns = [qn.strip()for qn in open('data/questions.txt')]
  aug = [qn.strip() for qn in open('data/augmented.txt')]
  all_qns = []
  for idx, qn in enumerate(orig_qns):
    all_qns.append(qn)
    if aug[idx] != qn:
      all_qns.append(aug[idx])
  print("Combined original questions and augmented questions")
  pickle.dump(all_qns, open("precompute/questions.pkl", 'wb'))

  qns = pickle.load(open("precompute/questions.pkl", 'rb'))
  documents = []
  for qn in qns:
    document = get_similar.preprocess_text(qn)
    if len(document) < 1:
      document = ['UNK']
    documents.append(document)

  print(f"Finished preprocessing {len(documents)} questions")
  pickle.dump(documents, open("precompute/documents.pkl", "wb"))
  print("Saved tokens to documents.pkl")
  documents = pickle.load(open("precompute/documents.pkl", "rb"))
  
  dct = corpora.Dictionary(documents)
  pickle.dump(dct, open("precompute/dct.pkl", 'wb'))
  dct.save('precompute/dct.dict')
  dct = corpora.Dictionary.load('precompute/dct.dict')
  
  corpus = [dct.doc2bow(doc) for doc in documents]
  pickle.dump(corpus, open("precompute/corpus.pkl", 'wb'))
  print("Corpus generated")

  tfidf = models.TfidfModel(corpus, smartirs='bfn')
  pickle.dump(tfidf, open("precompute/tfidf_model.pkl", 'wb'))
  corpus_tfidf = tfidf[corpus]
  pickle.dump(corpus_tfidf, open("precompute/corpus_tfidf.pkl", 'wb'))
  print("tfidf generated")

  index_temp = get_tmpfile("index")
  index = Similarity(index_temp, corpus_tfidf, num_features=len(dct), num_best=100)
  index.save("precompute/similarities.pkl")
  print("Similarity index saved")

  PIPE = subprocess.PIPE
  #NLU = subprocess.Popen(['rasa', 'train', '--data', ' nlu-train-data', '--fixed-model-name', 'model', '-vv', 'nlu'], stdout=PIPE, stderr=PIPE)
  NLU = subprocess.Popen(['rasa', 'train', 'nlu', '-u', 'nlu-train-data', '--config', 'config.yml', '--fixed-model-name', 'model'])
  NLU.wait()
  print("Rasa NLU trained")


if __name__=='__main__':
  main()
