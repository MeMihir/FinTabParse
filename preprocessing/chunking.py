from nltk.tokenize import word_tokenize

def perepare_chunk(table, row_chunk_no):
  def add_header(chunk):
    chunk.insert(0, table[0])
    return chunk
  
  chunk_vals = range(row_chunk_no[-1]+1)
  chunks = [[row for r, row in enumerate(table[1:]) if row_chunk_no[r]==x
             ] for x in chunk_vals]
  return list(map(add_header, chunks))

def get_table_chunks(table, chunk_size=512):
  header = table[0]
  header_tokens = len(word_tokenize(' '.join(table[0])))
  row_tokens = list(map(lambda row: len(word_tokenize(' '.join(row))), table[1:]))
  cum_row_tokens = [sum(row_tokens[0:x:1])//(chunk_size-header_tokens) 
    for x in range(0, len(row_tokens)+1)]
  row_chunk_no = cum_row_tokens[1:]
  # chunk_vals = range(row_chunk_no[-1]+1)
  # chunks = [[row for r, row in enumerate(table) if row_chunk_no[r]==x] for x in chunk_vals]
             
  return perepare_chunk(table, row_chunk_no)

def get_paragraphs_chunks(paragraph, chunk_size=512, overlap=16):
  tokenized = word_tokenize(paragraph)
  return [' '.join(tokenized[i:i+chunk_size]) for i in range(0, len(tokenized), chunk_size-overlap)]
  