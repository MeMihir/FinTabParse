from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def get_tfidf(table, paragraphs, header_rows, header_cols):
  table = np.array(table)
  para_links = np.empty(len(paragraphs))
  link_cols = []
  link_rows = []

  # print("Linked COLS")
  for col in header_cols:
    docs = table[len(header_rows)-1:,col]
    for para_indx, para in enumerate(paragraphs):
      docs[0] = para
      tfidf = TfidfVectorizer().fit_transform(docs)
      cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
      related_docs_indices = cosine_similarities.argsort()[::-1]
      link_col_indx = [idx for idx, val in enumerate(cosine_similarities) if val>0 and val<=0.9999999]
      link_cols.append([])
      for link_col in link_col_indx:
        link_cols[-1].append(related_docs_indices[link_col])
        # link_cols.append(related_docs_indices[link_col])
      # print(related_docs_indices, cosine_similarities, link_cols)
      # print(f"{para_indx}\t {link_cols}") 
      # print(f"\t {cosine_similarities}\n\n")

  # print("\n\n\nLinked ROWS")
  for row in header_rows:
    docs = table[len(header_cols)-1:,row]
    for para_indx, para in enumerate(paragraphs):
      docs[0] = para
      tfidf = TfidfVectorizer().fit_transform(docs)
      cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
      related_docs_indices = cosine_similarities.argsort()[::-1]
      link_row_indx = [idx for idx, val in enumerate(cosine_similarities) if val>0 and val<=0.9999999]
      # link_rows = []
      link_rows.append([])
      for link_row in link_row_indx:
        link_rows[-1].append(related_docs_indices[link_row])
        # link_rows.append(related_docs_indices[link_row])
      # print(related_docs_indices, cosine_similarities, link_rows)
      # print(f"{para_indx}\t {link_rows}")
      # print(f"\t {cosine_similarities}\n\n")
  return link_cols, link_rows