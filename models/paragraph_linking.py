from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def get_tfidf(table, paragraphs):
  table = np.array(table)
  para_links = np.empty(len(paragraphs))
  link_cells = []

  for i, para in enumerate(paragraphs):
    link_cells.append([])
    for j, row in enumerate(table):  
      docs = np.array([para])
      docs = np.append(docs, row)
      tfidf = TfidfVectorizer().fit_transform(docs)
      cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
      related_docs_indices = cosine_similarities.argsort()[::-1]
      link_col_indx = [idx for idx, val in enumerate(cosine_similarities) if val>0 and val<=0.9999999]
      for col in link_col_indx:
        link_cells[-1].append((j, col-1))
    
  return link_cells