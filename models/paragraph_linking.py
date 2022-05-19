from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from re import match as re_match

def is_number_regex(s):
    """ Returns True is string is a number. """
    if re_match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True


def get_tfidf(table, paragraphs, base_thresh=0.1, dynamic_thresh=0.5):
  table = np.array(table)
  para_links = np.empty(len(paragraphs))
  link_cells = []

  for r, row in enumerate(table):
    for c, cell in enumerate(row):
      if(is_number_regex(cell)):
        continue
      docs = np.array(paragraphs)
      docs = np.insert(docs, 0, cell)
      tfidf = TfidfVectorizer().fit_transform(docs)
      cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
      related_docs_indices = cosine_similarities.argsort()[::-1]
      max_sim = max(cosine_similarities[1:])
      link_para_indx = [idx for idx, val in enumerate(cosine_similarities[1:]) if 
                        val > max(base_thresh, max_sim*dynamic_thresh)
                        ]
      link_cells.append({'r': r, 'c': c, 'paras': [], 'cell': cell})
      # print(cosine_similarities)
      for para in link_para_indx:
        link_cells[-1]["paras"].append(f'link_{para}')
    
  return link_cells