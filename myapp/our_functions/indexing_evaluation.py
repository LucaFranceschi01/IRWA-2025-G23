# File created using copilot, original code is from the part2 notebook

import collections
import string
import re
import math
from collections import defaultdict
from array import array

import pandas as pd
import numpy as np
import numpy.linalg as la
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# --- NLTK bootstrap (optional) ---
def ensure_nltk(download: bool = False):
    """
    Ensure NLTK resources exist. Set download=True to attempt a silent download.
    """
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except Exception:
        if download:
            import nltk
            nltk.download('punkt', quiet=True)
    try:
        stopwords.words('english')
    except LookupError:
        if download:
            import nltk
            nltk.download('stopwords', quiet=True)


# --- Preprocessing ---
def preprocess_text(text, stemmer=None, stop_words=None, translator=None):
    """
    Lowercase, remove punctuation, unidecode, tokenize, remove stopwords, stem, remove short tokens.
    Returns a list of tokens.
    """
    translator = translator or str.maketrans('', '', string.punctuation)
    stemmer = stemmer or PorterStemmer()
    stop_words = stop_words or set(stopwords.words('english'))
    if not isinstance(text, str):
        text = '' if text is None else str(text)
    text = unidecode(text.lower().translate(translator))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 2]
    return tokens


def create_document_for_each_row(row, text_columns):
    """
    Concatenate specified text fields for a row into a single string.
    """
    text_fields = [str(row[col]) for col in text_columns if col in row and pd.notnull(row[col])]
    document = ' '.join([f for f in text_fields if f != 'nan' and f != '']).lower()
    return document


# --- Try to reuse the Part-1 preprocessor if available ---
try:
    from myapp.our_functions.data_preparation import TextPreprocessor  # when running from repo root
except Exception:
    try:
        from data_preparation import TextPreprocessor  # when running from project_progress folder
    except Exception:
        TextPreprocessor = None

_PREPROCESSOR = TextPreprocessor() if TextPreprocessor else None

def _preprocess_tokens(text: str):
    """
    Prefer Part-1 TextPreprocessor (returns space-joined string) and convert to tokens.
    Fallback to local preprocess_text (returns tokens already).
    """
    if _PREPROCESSOR is not None:
        if not isinstance(text, str):
            text = '' if text is None else str(text)
        return [t for t in _PREPROCESSOR(text).split() if t]
    # fallback to the locally defined pipeline
    return preprocess_text(text)

# --- Inverted index + AND search (small edit to use unified preprocessor) ---
def build_inverted_index(data: pd.DataFrame, text_columns):
    """
    Build an inverted index mapping term -> [pid, ...] with deduplicated PIDs per term.
    Text is preprocessed (tokenized, stemmed).
    """
    inverted_index = defaultdict(list)
    for _, row in data.iterrows():
        doc_id = row['pid']
        row_text = create_document_for_each_row(row, text_columns)

        # Preprocess and tokenize
        """ row_text = ' '.join(preprocess_text(row_text))
        tokens = re.findall(r'\b\w+\b', row_text) """
        tokens = _preprocess_tokens(row_text)
        tokens = re.findall(r'\b\w+\b', ' '.join(tokens))

        already_seen_terms = set()
        for term in tokens:
            if term not in already_seen_terms:
                inverted_index[term].append(doc_id)
                already_seen_terms.add(term)
    return dict(inverted_index)


def conjunctive_search(query: str, inverted_index: dict, preprocess: bool = False):
    """
    AND query. Returns list of documents containing ALL query terms.
    Set preprocess=True to apply the same preprocessing to the query.
    """
    terms = preprocess_text(query) if preprocess else query.split(' ')
    intersection = set()
    for t in terms:
        if t not in inverted_index:
            return []
        intersection = set(inverted_index[t]) if not intersection else intersection.intersection(inverted_index[t])
    return list(intersection)


# --- TF-IDF index + ranking ---
def create_index_tfidf(data: pd.DataFrame, columns: list):
    """
    Build positional inverted index and compute tf, df, idf over the provided columns.
    
    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of column names to index.
        
    Returns:
        tuple: (index, tf, df, idf)
    """
    index = defaultdict(list)
    tf = defaultdict(list)
    df = defaultdict(int)
    idf = defaultdict(float)
    N = len(data.index)

    for _, row in data.iterrows():
        page_id = row['pid']
        joined = ' '.join(str(v) for v in row[columns] if pd.notnull(v))
        ###terms = preprocess_text(joined)
        terms = _preprocess_tokens(joined)

        current_page_index = {}
        for position, term in enumerate(terms):
            try:
                current_page_index[term][1].append(position)
            except KeyError:
                current_page_index[term] = [page_id, array('I', [position])]

        # Normalize TF
        norm = 0.0
        for term, posting in current_page_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm) if norm > 0 else 1.0

        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4))
            df[term] += 1

        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)
            
    # Compute IDF
    for term in df:
        idf[term] = np.round(np.log(float(N) / df[term]), 4)

    return index, tf, df, idf


def rank_documents(terms, docs, index, tf, idf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights.

    Args:
        terms (list): List of query terms.
        docs (list): List of documents to rank (matching the query).
        index (dict): Inverted index data structure.
        tf (dict): Term frequencies.
        idf (dict): Inverted document frequencies.

    Returns:
        list: List of [score, doc_id] sorted by score descending.
    """
    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    # Compute the norm for the query tf
    query_terms_count = collections.Counter(terms)
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        # Compute tf*idf (normalize TF as done with documents)
        query_vector[termIndex] = query_terms_count[term] / query_norm * idf[term]

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # tf[term][doc_index] will contain the tf of the term 'term' in the doc
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    # Calculate the score of each doc
    # Compute the cosine similarity between query_vector and each doc_vector
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)

    return doc_scores

def search_tf_idf(query, index, tf, idf):
    """
    Search for documents matching the query using TF-IDF and return ranked results.
    
    Args:
        query (str): The search query.
        index (dict): Inverted index.
        tf (dict): Term frequencies.
        idf (dict): Inverse document frequencies.
        
    Returns:
        list: Ranked list of documents [score, doc_id].
    """
    query_tokens = _preprocess_tokens(query)
    docs = set()
    for term in query_tokens:
        try:
            # store in term_docs the ids of the docs that contain 'term'
            term_docs = [posting[0] for posting in index[term]]
            docs = docs.union(set(term_docs))
        except KeyError:
            # term is not in index
            pass
            
    docs = list(docs)
    ranked_docs = rank_documents(query_tokens, docs, index, tf, idf)
    return ranked_docs



# --- EVALUATION METRICS --- #

def precision_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, k: int) -> float:
    """
    Calculate Precision@K.
    Precision is the share of retrieved documents that are relevant.
    """
    retrieved = ranked_docs[ranked_docs['rank'] <= k]['pid'].values
    relevant = val_data[val_data['labels'] == 1]['pid'].values
    not_relevant = val_data[val_data['labels'] == 0]['pid'].values

    tp = len(set(retrieved) & set(relevant))
    fp = len(set(retrieved) & set(not_relevant))

    if tp == 0 and fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, k: int) -> float:
    """
    Calculate Recall@K.
    Recall is the share of relevant documents that are retrieved.
    """
    retrieved = ranked_docs[ranked_docs['rank'] <= k]['pid'].values
    not_retrieved = ranked_docs[ranked_docs['rank'] > k]['pid'].values
    relevant = val_data[val_data['labels'] == 1]['pid'].values

    tp = len(set(retrieved) & set(relevant))
    fn = len(set(not_retrieved) & set(relevant))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def average_precision_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, n: int) -> float:
    """
    Calculate Average Precision@K.
    Average Precision is the average of precision scores at each rank position where a relevant document is found.
    """
    gtp = 0
    prec_at_i_list = []
    prev_precision = -1
    
    for k in range(1, n+1):
        retrieved = ranked_docs[ranked_docs['rank'] <= k]['pid'].values
        relevant = val_data[val_data['labels'] == 1]['pid'].values

        # always increases, last value is saved
        gtp = len(set(retrieved) & set(relevant))

        current_precision = precision_at_k(ranked_docs, val_data, k)

        if current_precision > prev_precision:
            prec_at_i_list.append(current_precision)
        else:
            prec_at_i_list.append(0)
        
        prev_precision = current_precision

    if gtp == 0:
        return 0.0

    return float((1 / gtp) * np.sum(prec_at_i_list))

def f1_score_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, k: int) -> float:
    """
    Calculate F1-score@K.
    F1-score is the harmonic mean of precision and recall.
    """
    prec = precision_at_k(ranked_docs, val_data, k)
    rec = recall_at_k(ranked_docs, val_data, k)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def ndcg_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)@K.
    Measures the graded relevance of the retrieved documents.
    """
    retrieved = ranked_docs[ranked_docs['rank'] <= k]['pid'].values
    relevant = val_data[val_data['labels'] == 1]['pid'].values

    # binary relevance even if we could do more levels of relevance
    relevance_scores = [1 if doc in relevant else 0 for doc in retrieved]
    ideal_list = sorted(relevance_scores, reverse=True)
    
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal_dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_list))
    
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def rr_at_k(ranked_docs: pd.DataFrame, val_data: pd.DataFrame, k: int) -> float:
    """
    Calculate Reciprocal Rank@K.
    """
    merged = val_data.merge(ranked_docs, on='pid', how='left')
    # Filter only relevant docs that were retrieved
    relevant_ranks = merged[merged['labels'] == 1]['rank']
    
    if relevant_ranks.empty:
        return 0.0
        
    sCorrect_Ri = relevant_ranks.min()
    
    if pd.isna(sCorrect_Ri) or sCorrect_Ri > k:
        return 0.0
    
    return 1 / float(sCorrect_Ri)


def mean_average_precision(results: dict) -> float:
    """
    Calculate Mean Average Precision (MAP).
    MAP is the mean of average precision scores across multiple queries.
    """
    ap_scores = []
    for _, query_result in results.items():
        ap_scores.append(query_result['AveragePrecision@K'])

    return np.mean(ap_scores)


def mean_reciprocal_rank(results: dict) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    MRR is the average of the reciprocal ranks of the first relevant document across multiple queries.
    """
    rr_scores = []
    for _, query_result in results.items():
        rr_scores.append(query_result['RR@K'])

    return np.mean(rr_scores)