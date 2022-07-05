import pandas as pd
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import spacy

def tokenize(text):
    text = re.sub(r'[^A-Za-z]',' ',text.upper())
    tokenized_words = word_tokenize(text)
    return tokenized_words

def remove_stopwords(words, stop_words):
    return [x for x in words if x not in stop_words]

def positive_score(words, positive_dict):
    return len([x for x in words if x.lower() in positive_dict])

def negative_score(words, negative_dict):
    return len([x for x in words if x.lower() in negative_dict])

def polarity(positive_score, negative_score):
    return (positive_score - negative_score)/((positive_score + negative_score)+ 0.000001)

def subjectivity(positive_score, negative_score, num_words):
    return (positive_score+negative_score)/(num_words+ 0.000001)

def fog_index(average_sentence_length, percentage_complexwords):
    return 0.4*(average_sentence_length + percentage_complexwords)

def syllable_count(words):
    k = []
    for word in words:
        count = 0
        v = []
        d = {}
        if(word[-2:] == 'ES' or word[-2:] == 'ED'):
            count = count - 1
        for i in word:
            if i in "AEIOU":
                v.append(i)
                d[i] = d.get(i,0)+1
        for i in d:
            count = count + d[i]
        k.append(count)
            
    return k

def complex_word_count(syllable_count):
    count = 0
    for i in syllable_count:
        if i>2:
            count += 1
    return count

def char_count(words):
    count = 0
    for word in words:
        count += len(word)
    return 

def transform(text, stopwords):
    text = tokenize(text)
    return remove_stopwords(text, stopwords)

data = pd.read_excel('input.xlsx', index_col=0)
data['URL']

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"
}

articles = pd.DataFrame(columns=['Title', 'Text'])

for index, url in enumerate(data['URL']):
    html_text = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html_text, 'lxml')
    title = soup.find('h1', class_='entry-title').text
    article_text = soup.find('div', class_='td-post-content').text
    
    articles = articles.append({'Title': title, 'Text': article_text}, ignore_index=True)
    
    with open(f'Extracted Articles/{index+1}.txt', 'w+', errors='ignore') as f:
        f.write(f'{title}\n')
        f.write(article_text)

articles.to_excel('Articles.xlsx')

articles = pd.read_excel('Articles.xlsx', index_col=0)
articles

df = pd.concat([data, articles['Text']], axis=1)
df

with open('MasterDictionary/positive-words.txt' ,'r') as f:
    positive_dict = f.read()

with open('MasterDictionary/negative-words.txt' ,'r') as f:
    negative_dict = f.read()

positive_dict = positive_dict.split('\n')
negative_dict = negative_dict.split('\n')

with open('StopWords/StopWords_Auditor.txt', 'r') as f1, open('StopWords/StopWords_Currencies.txt', 'r') as f2, \
open('StopWords/StopWords_DatesandNumbers.txt', 'r') as f3, open('StopWords/StopWords_Generic.txt', 'r') as f4, \
open('StopWords/StopWords_GenericLong.txt', 'r') as f5, open('StopWords/StopWords_Geographic.txt', 'r') as f6, \
open('StopWords/StopWords_Names.txt', 'r') as f7:
    stopwords1 = f1.read()
    stopwords2 = f2.read()
    stopwords3 = f3.read()
    stopwords4 = f4.read()
    stopwords5 = f5.read()
    stopwords6 = f6.read()
    stopwords7 = f7.read()

stopwords1 = stopwords1.split('\n')
stopwords2 = stopwords2.replace('|', '\n').replace(' ','').split('\n')
stopwords3 = stopwords3.replace('|', '\n').replace(' ','').split('\n')
stopwords4 = stopwords4.split('\n')
stopwords5 = stopwords5.split('\n')
stopwords6 = stopwords6.replace('|', '\n').replace(' ', '').split('\n')
stopwords7 = stopwords7.split('\n')
stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4 + stopwords5 + stopwords6 + stopwords7

# Cleaning using Stopwords Lists
df['Transformed_text'] = df['Text'].apply(lambda tx: transform(tx, stopwords))

# Calculating Variables
df['Positive Score'] = df['Transformed_text'].apply(lambda tx: positive_score(tx, positive_dict))

df['Negative Score'] = df['Transformed_text'].apply(lambda tx: negative_score(tx, negative_dict))

df['Polarity Score'] = (df['Positive Score']-df['Negative Score'])/((df['Positive Score']-df['Negative Score'])+0.000001)

df['Subjectivity Score'] = (df['Positive Score']-df['Negative Score'])/(len(df['Transformed_text']) + 0.000001)

df['Word Count'] = df['Transformed_text'].apply(lambda x: len(str(x).split()))

df['Syllable per Word'] = df['Transformed_text'].apply(lambda words: syllable_count(words))

df['Complex Word Count'] = df['Syllable per Word'].apply(lambda syllable: complex_word_count(syllable))

df['sentence length'] = np.nan
df['Avg Sentence Length'] = np.nan
df['Percentage of Complex Words'] = np.nan
df['Fog Index'] = np.nan

for i in range(0,len(df)):
    df['sentence length'][i]  =   len(nltk.sent_tokenize(df['Text'][i]))
    df['Avg Sentence Length'][i] = len(nltk.word_tokenize(df['Text'][i]))/df['sentence length'][i]
    df['Percentage of Complex Words'][i] = df['Complex Word Count'][i]/df['Word Count'][i] 
    df['Fog Index'][i] = 0.4 * (df['Avg Sentence Length'][i] + df['Percentage of Complex Words'][i])


df['Avg Number of Words Per Sentence'] = np.nan

for i in range(0,len(df)):
    df['Avg Number of Words Per Sentence'][i] = df['Word Count'][i]/len(nltk.sent_tokenize(df['Text'][i]))

df['chara_count'] = df['Transformed_text'].apply(lambda words: char_count(words))
df['Avg Word Length'] = np.nan

for i in range(0,len(df)):
    df['Avg Word Length'][i] = df['chara_count'][i]/df['Word Count'][i]

# Loading Spacy api for personal pronouns
nlp = spacy.load('en_core_web_sm')

df['Personal Pronouns'] = np.nan

for i in range(len(df)):
    doc = nlp(df['Text'][i])
    tok = []
    for token in doc:
        if token.pos_ == 'PRON':
            tok.append(token)
        
    df['Personal Pronouns'][i] = tok

df = df.drop(columns=['sentence length', 'chara_count', 'Text', 'Transformed_text'])
df = df[['URL_ID', 'URL', 'Positive Score', 'Negative Score', 'Polarity Score', 'Subjectivity Score', 'Avg Sentence Length', \
    'Percentage of Complex Words', 'Fog Index', 'Avg Number of Words Per Sentence', 'Complex Word Count', 'Word Count', \
   'Syllable per Word', 'Personal Pronouns', 'Avg Word Length']]

df.to_excel('Output Data Structure.xlsx')