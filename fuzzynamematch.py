import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import locale
locale.setlocale(locale.LC_ALL, 'tr_TR')

nltk.download('punkt')
nltk.download('stopwords')

import requests
from bs4 import BeautifulSoup



url = "https://www.kap.org.tr/tr/bist-sirketler"
ret = requests.get(url)
soup = BeautifulSoup(ret.content, 'html.parser')
divs = soup.find_all('div', class_='column-type7 wmargin')
names = [f.find('a').text for div in divs for f in div.find_all('div', class_='comp-cell _14 vtable')]
names

df = pd.DataFrame(names, columns=['company_name'])
df.head()

def preprocess(text):
    lower_map = {
        ord(u'I'): u'ı',
        ord(u'İ'): u'i',
        }    
    text = re.sub(r'\b\w+\d+\w*\b', '', text)  # Remove words containing numbers
    text = re.sub('[^a-zA-Z\sİŞÜĞÇÖ]', '', text)  # Remove non-alphabetic characters
    text = text.translate(lower_map).lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize
    tokens = [token for token in tokens if token not in stopwords.words('turkish')]  # Remove stopwords
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]  # Stemming
    return ' '.join(tokens)

df['preprocessed_name'] = df['company_name'].apply(preprocess)
df.head()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed_name'])
tfidf_matrix

def get_potential_matches(query, df, fuzz_threshold=70):
    preprocessed_query = preprocess(query)
    choices = df['preprocessed_name'].tolist()
    
    matches = [
        (i, fuzz.token_set_ratio(preprocessed_query, choice))
        for i, choice in enumerate(choices)
    ]
    
    matched_indices = [i for i, (_, score) in enumerate(matches) if score >= fuzz_threshold]
    return matched_indices

def search_company(query, df, tfidf_matrix, fuzz_threshold=70, score_threshold=0.5):
    matched_indices = get_potential_matches(query, df, fuzz_threshold)
    
    if not matched_indices:
        return pd.DataFrame()

    query_vector = vectorizer.transform([preprocess(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[matched_indices])
    similarities = cosine_similarities.flatten()
    
    filtered_indices = np.where(similarities > score_threshold)[0]
    
    if not filtered_indices.size:
        return pd.DataFrame()

    matched_df = df.iloc[matched_indices].iloc[filtered_indices].copy()
    matched_df['similarity_score'] = similarities[filtered_indices]
    matched_df = matched_df.sort_values(by='similarity_score', ascending=False)
    return matched_df


query = "DEVA HOLD."
results = search_company(query, df, tfidf_matrix)
display(results)
