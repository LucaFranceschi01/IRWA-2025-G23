# File created using copilot, original code is from the part1 notebook

import os, string
import numpy as np
import pandas as pd
from collections import Counter
import re

from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load data from json file
def load_json_dataset(path=None):
    """
    Load dataset JSON. If path is None, tries ../../data/fashion_products_dataset.json relative to cwd.
    Returns a pandas.DataFrame.
    """
    if path is None:
        data_path = os.path.join(os.getcwd(), '../../data')
        path = os.path.join(data_path, 'fashion_products_dataset.json')
    return pd.read_json(path)

def build_text_preprocessor(stop_words=None, stemmer=None):
    """
    Returns a preprocess_text(text) function configured with given stop_words and stemmer.
    """
    translator = str.maketrans('', '', string.punctuation)
    if stemmer is None:
        stemmer = PorterStemmer()
    if stop_words is None:
        stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = text.translate(translator)
        text = unidecode(text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        tokens = [stemmer.stem(w) for w in tokens]
        tokens = [w for w in tokens if len(w) > 2]
        return " ".join(tokens)

    return preprocess_text

class TextPreprocessor:
    """Handles text preprocessing with stemming and stopword removal."""
    
    def __init__(self, stop_words=None, stemmer=None):
        self.translator = str.maketrans('', '', string.punctuation)
        self.stemmer = stemmer or PorterStemmer()
        self.stop_words = stop_words or set(stopwords.words('english'))
    
    def __call__(self, text):
        """Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, stem."""
        if not isinstance(text, str):
            return ""
        
        text = unidecode(text.lower().translate(self.translator))
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words]
        tokens = [self.stemmer.stem(w) for w in tokens]
        tokens = [w for w in tokens if len(w) > 2]
        return " ".join(tokens)


def select_relevant_columns(df, relevant_columns=None):
    """
    Keep only relevant columns that exist in df.
    """
    if relevant_columns is None:
        relevant_columns = [
            'pid', 'title', 'description', 'brand',
            'category', 'sub_category', 'product_details',
            'seller', 'out_of_stock', 'selling_price',
            'discount', 'actual_price', 'average_rating', 'url'
        ]
    available = [f for f in relevant_columns if f in df.columns]
    return df[available].copy()

def expand_product_details(df, top_n=25):
    """
    Extract product_details (list of dicts per product) to separate columns for the top_n most common keys.
    Drops original product_details column.
    """
    details = [{k: v for d in (lst or []) for k, v in d.items()} for lst in df.get("product_details", [])]
    cnt = Counter(k for d in details for k in d)
    top_keys = [k for k,_ in cnt.most_common(max(top_n, 0))]
    # Build DataFrame aligned to rows of df
    sub = pd.DataFrame(details).reindex(columns=top_keys)
    norm = lambda s: re.sub(r'\W+', '_', s.strip().lower())
    sub.columns = [f"detail_{norm(c)}" for c in sub.columns]
    # Join and drop original
    out = df.join(sub)
    if "product_details" in out.columns:
        out = out.drop(columns=["product_details"])
    return out

def clean_metadata_fields(df):
    """
    Normalize metadata fields: lowercase, remove punctuation, strip whitespace.
    Returns a copy with cleaned columns.
    """
    translator = str.maketrans('', '', string.punctuation)
    metadata_cols = ['category', 'sub_category', 'brand', 'seller']
    detail_cols = [col for col in df.columns if col.startswith('detail_')]
    
    for col in metadata_cols + detail_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.lower().translate(translator).strip() if isinstance(x, str) else ""
            )
    
    return df

def convert_numeric_fields(df):
    """
    Convert out_of_stock, selling_price, discount, actual_price, average_rating to numeric types where possible.
    Returns a copy with converted columns.
    """
    df = df.copy()
    
    # Boolean field
    if "out_of_stock" in df.columns:
        df["out_of_stock"] = df["out_of_stock"].astype(int)
    
    # Price fields (remove commas, convert to float)
    for price_col in ["selling_price", "actual_price"]:
        if price_col in df.columns:
            df[price_col] = (df[price_col].astype(str)
                            .str.replace(',', '', regex=False)
                            .replace(['', 'nan'], np.nan)
                            .astype(float))
    
    # Discount field (remove %, 'off', spaces)
    if "discount" in df.columns:
        df["discount"] = (df["discount"].astype(str)
                         .str.replace(r'[%\s]|off', '', regex=True)
                         .replace(['', 'nan'], np.nan)
                         .astype(float))
    
    # Rating field
    if "average_rating" in df.columns:
        df["average_rating"] = df["average_rating"].replace('', np.nan).astype(float)
    
    return df

# -- MAIN FUNCTION --
def prepare_data(
    json_path=None,
    export_csv=True,
    csv_path='fashion_products_cleaned.csv',
    top_detail_n=25,
    download_nltk=False
):
    """
    Full pipeline:
    - load JSON
    - select relevant columns
    - expand product_details into top_detail_n columns
    - clean metadata fields
    - preprocess title/description text (stem + stopwords)
    - convert numeric fields
    - export cleaned CSV (optional)
    Returns cleaned DataFrame.
    """
    if download_nltk:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    # Load and filter data
    df = load_json_dataset(json_path)
    df = select_relevant_columns(df)

    # Feature engineering
    df = expand_product_details(df, top_n=top_detail_n)
    df = clean_metadata_fields(df)

    # Text processing
    preprocessor = TextPreprocessor()
    for text_col in ['title', 'description']:
        if text_col in df.columns:
            df[text_col] = df[text_col].apply(preprocessor)

    df = convert_numeric_fields(df)

    if export_csv:
        df.to_csv(csv_path, index=False)

    return df



if __name__ == "__main__":
    # Execute the data preparation pipeline
    df = prepare_data(
        json_path=None,
        export_csv=True,
        csv_path='../../data/fashion_products_cleaned_script.csv',
        top_detail_n=25,
        download_nltk=False
    )
    print(f"✓ Data preparation complete!")
    print(f"  Shape: {df.shape}")
    print(f"  Output: data/fashion_products_cleaned_script.csv")
