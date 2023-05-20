from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.error
import requests as r
import pandas as pd
import os
import sys
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
nltk.download('punkt')

#Tokenizer
def tokenizer(text):
	lower_text = text.lower()
	raw_data = [lower_text]
	word_tok = [word_tokenize(i) for i in raw_data]
	clean_text = []

	for words in word_tok:
		for w in words:
			res = re.sub(r'[^\w\s\[\]]','',w)
			if res != '':
				clean_text.append(res)
	return clean_text

#LOADING STOP WORDS.
stopword_list = []
folder_path = r"C:\Users\shiba\OneDrive\Desktop\BlackCoffer\StopWords"
filenames = os.listdir(folder_path)
for filename in filenames:
    # check if the file is a text file
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            contents = file.read().lower()
            words = re.sub(r'[^\w\s]', '', contents).split()
            stopword_list += words

#LOADING POSITIVE AND NEGATIVE WORDS.
folder_path = r"C:\Users\shiba\OneDrive\Desktop\BlackCoffer\MasterDictionary"

negativewords_list = []
positivewords_list = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the current file is a text file
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = re.sub(r'[^\w\s]', '', line).strip().split()
                if filename == 'negative-words.txt':
                    negativewords_list += words
                elif filename == 'positive-words.txt':
                    positivewords_list += words

#CALCULATING POSITIVE SCORE
def positive_score(text):
	num_pos_words = 0
	raw_token = tokenizer(text)
	for word in raw_token:
		if word in positivewords_list:
			num_pos_words += 1
	sum_pos = num_pos_words
	return sum_pos

#CALCULATING NEGATIVE SCORE
def negative_score(text):
	num_neg_words = 0
	raw_token = tokenizer(text)
	for word in raw_token:
		if word in negativewords_list:
			num_neg_words += 1
	sum_neg = num_neg_words
	return sum_neg

#CALCULATING POLARITY SCORE
def get_polarity_score(positiveScore, negativeScore):
	pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
	return pol_score

#CALCULATING SUBJECTIVITY SCORE.
def get_subjectivity_score(positiveScore, negativeScore, total_clean_word):
    subjectivity_score = (positiveScore + negativeScore) / (total_clean_word + 0.000001)
    return subjectivity_score

#CALCULATING AVERAGE SENTENCE LENGTH
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)

#CALCULATING PERCENTAGE OF COMPLEX WORD
def percentage_complex_word(text):
    tokens = tokenizer(text)
    complexWord = 0
    complex_word_percentage = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    if len(tokens) != 0:
        complex_word_percentage = complexWord/len(tokens)
    
    return complex_word_percentage

#COUNTING AVG NUMBER OF WORDS PER SENTENCE.
def average_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_words = 0
    for sentence in sentences:
        total_words += len(word_tokenize(sentence))
    average_words = total_words / len(sentences)
    return average_words

#CALCULATING FOG INDEX
def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex

#WORD COUNT
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)
#COUNTING COMPLEX WORDS
def complex_word_count(text):
    tokens = tokenizer(text)
    complexWord = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    return complexWord

#COUNTING SYLLABLE PER WORD.
cmu_dict = nltk.corpus.cmudict.dict()

def syllables_per_word(word):
    if word.lower() in cmu_dict:
        syllables = [len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]][0]
        return syllables
    else:
        return 0
def average_syllables_per_word(text):
    words = word_tokenize(text)
    total_syllables = 0
    for word in words:
        total_syllables += syllables_per_word(word)
    average_syllables = total_syllables / len(words)
    return average_syllables

#COUNTING PERSONAL PRONOUNS
def frequency_of_personal_pronouns(text):
    words = tokenizer(text)
    words = [word for word in words if word.lower() not in stopword_list]
    # tag words as personal pronouns or not
    personal_pronouns = ["I", "me", "you", "he", "him", "she", "her", "we", "us", "they", "them"]
    tagged_words = nltk.pos_tag(words)
    personal_pronoun_count = 0
    for word, pos in tagged_words:
        if word.lower() in personal_pronouns:
            personal_pronoun_count += 1
    # calculate frequency of personal pronouns
    frequency = personal_pronoun_count / len(words)
    return frequency

#CALCULATING AVG WORD LENGTH
def avg_word_length(text):
    words = tokenizer(text)
    total_length = 0
    for word in words:
        total_length += len(word)
    avg_length = total_length / len(words)
    return avg_length

#DATA EXTRACTION FROM URL
# def data_extract_from_url(url):
#     try:
#         html = urlopen(url)
#     except urllib.error.HTTPError as error:
#         print(f"An error occurred for {url}: {error}")
#         continue
#     soup = BeautifulSoup(html, "html.parser")
#     title = soup.find("title").text
#     all_links = soup.findAll('div', {'class': 'td-container'})
#     str_cells = str(all_links)
#     cleartext = title + BeautifulSoup(str_cells, "html.parser").get_text()
#     return cleartext

#MAIN PART..
df1 = pd.read_excel("C:\\Users\\shiba\\OneDrive\\Desktop\\BlackCoffer\\Input.xlsx")
df2 = pd.read_excel("C:\\Users\\shiba\\OneDrive\\Desktop\\BlackCoffer\\Output Data Structure.xlsx", sheet_name = "Sheet1")
count = 0
for index, row in df1.iterrows():
    url = row['URL']
    if url == "":
        break
    try:
        html = urlopen(url)
    except urllib.error.HTTPError as error:
        print(f"An error occurred for {url}: {error}")
        continue
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title").text
    all_links = soup.findAll('div', {'class': 'td-container'})
    str_cells = str(all_links)
    cleartext = title + BeautifulSoup(str_cells, "html.parser").get_text()

    t = re.sub(r'[^a-zA-Z0-9\s],\(\)\*', '', cleartext)
    total_clean_t = len(tokenizer(t))
    df2.loc[count, 'POSITIVE SCORE'] = p = positive_score(t)
    df2.loc[count, 'NEGATIVE SCORE'] = n = negative_score(t)
    df2.loc[count, 'POLARITY SCORE'] = get_polarity_score(p, n)
    df2.loc[count, 'SUBJECTIVITY SCORE'] = get_subjectivity_score(p, n, total_clean_t)
    df2.loc[count, 'AVG SENTENCE LENGTH'] = asl = average_sentence_length(t)
    df2.loc[count, 'PERCENTAGE OF COMPLEX WORDS'] = pcw = percentage_complex_word(t)
    df2.loc[count, 'FOG INDEX'] = fog_index(asl, pcw)
    df2.loc[count, 'AVG NUMBER OF WORDS PER SENTENCE'] = average_words_per_sentence(t)
    df2.loc[count, 'COMPLEX WORD COUNT'] = complex_word_count(t)
    df2.loc[count, 'WORD COUNT'] = total_word_count(t)
    df2.loc[count, 'SYLLABLE PER WORD'] = average_syllables_per_word(t)
    df2.loc[count, 'PERSONAL PRONOUNS'] = frequency_of_personal_pronouns(t)
    df2.loc[count, 'AVG WORD LENGTH'] = avg_word_length(t)

    count += 1


df2.to_excel("C:\\Users\\shiba\\OneDrive\\Desktop\\BlackCoffer\\Output Data Structure.xlsx", sheet_name = "Sheet1", index = False, engine = 'openpyxl',)