import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict, Tuple

from myapp.our_functions.data_preparation import prepare_data

def our_load_corpus(path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(path)
    #df_cleaned = our_build_clean_corpus(df)
    df_cleaned = prepare_data(df=df, export_csv=False, json_path=None, download_nltk=False)
    return df, df_cleaned

def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = pd.read_json(path)
    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        corpus[doc.pid] = doc
    return corpus

