#!/usr/bin/env python
# coding: utf-8

# In[243]:


import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
from nltk.corpus import brown
import requests
from bs4 import BeautifulSoup
nltk.download('popular')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[2]:


# Getting the HTML 
r = requests.get('https://archive.org/stream/fegmcfeggerson_gmail_4731/473%20%281%29_djvu.txt')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print(html[:1000])


# In[95]:


# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

# Printing out text between characters 9225 and 17345, which is only the document text

text = text.replace('\INU LIN DALE ', 'Ainulindale')
text = text.replace('V alar', 'Valar')
text = text.replace('liuvatar', 'Iluvatar')
text = text.replace('Flelcaraxe', 'Helcaraxe')
text = text.replace('Ore-host', 'Orc-host')
text = text.replace("■WWW. TedNaSITlWf.com ESI la Led to the Walla — Copyright 3“ Ted Nasmith .Ml rights reserved.", "")
text = text.replace('Ores', 'Orcs')
text = text.replace('F inrod', 'Finrod')
text = text.replace('Beleglt Slain -a»wkhi 6 1M Ntimith. All ngbU n*rwd. ', '')
text = text.replace('Nam i HTn Hurin', 'Narn i Hin Hurin')
text = text.replace('MTm', 'Mim')
text = text.replace('Thprp is hlnnrl on thp hill-tnn ', 'There is blood on the hill-top')
text = text.replace('nnfi amnnn thpm Raid', 'one among them said:')
text = text.replace('KhTm', 'Khim')
text = text.replace('Flear', 'Hear')
text = text.replace('Ore-band', 'Orc-band')
text = text.replace('NulukkizdTn', 'Nulukkizdin')
text = text.replace('Tlnuviel', 'Tinuviel')
text = text.replace('WWW.TetlNasmith.com Tuor U l*d by the Swans lo Vinyamar - CupynthtCT**! Numltti AD i ','')
text = text.replace('Morqoth', 'Morgoth')
text = text.replace('Ore', 'Orc')
text = text.replace('Shadow/', 'Shadow')
text = text.replace('andmthe', 'and the')
text = text.replace('www.TedNasmith.com ', '')
text = text.replace('Maglor Coal* a Silmaril into the Sea - Copyright © Ted Nasmith. AH rights rnentd.', '')
text = text.replace('ofVingilotto', 'of Vingilot to')
text = text.replace('inArmenelos', 'in Armenelos')
text = text.replace('Tlrion', 'Tirion')
text = text.replace('Condor', 'Gondor')



# Entire Book
book = text[11186:722336]

# Ainulindale
ainulindale = book[0:20310] 

# Valaquenta
valaquenta = book[20315:38505]

# Of The Beginning Of Days
ch1 = book[38563:59452]

# Of Aule And Yavanna
ch2 = book[59457:69920]

# Of The Coming Of The Elves And The Captivity Of Melkor
ch3 = book[69925:91210]

# Of Thingol And Melian
ch4 = book[91215:94655]

# Of Eldamar and the Princes of the Eldalie 
ch5 = book[94661:109186]

# Of Feanor and the Unchaining of Melkor 
ch6 = book[109191:118610]

# Of The Silmarils And The Unrest Of The Noldor
ch7 = book[118616:133480]

# Of the Darkening of Valinor 
ch8 = book[133485:144295]

# Of the Flight of the Noldor 
ch9 = book[144300:178590]

# Of the Sindar 
ch10 = book[178596:195710]

# Of the Sun and Moon and the Hiding of Valinor 
ch11 = book[195716:208710]

# Of Men
ch12 = book[208715:214924]

# Of the Return of the Noldor 
ch13 = book[214929:245814]

# Of Beleriand and Its Realms 
ch14 = book[245819:263938]

# Of the Noldor in Beleriand 
ch15 = book[263944:277760]

# Of Maeglin
ch16 = book[277765:300251]

# Of the Coming of Men into the West
ch17 = book[300257:325040]

# Of the Ruin of Beleriand and the Fall of Fingolfin 
ch18 = book[325045:355008]

# Of Beren and Luthien 
ch19 = book[355014:422465]

# Of the Fifth Battle: Nirnaeth Arnoediad 
ch20 = book[422471:447950]

# Of Turin Turambar
ch21 = book[447956:523704]

# Of the Ruin of Doriath 
ch22 = book[523709:550278]

# Of Tuor and the Fall of Gondolin
ch23 = book[550284:569583]

# Of the Voyage of Earendil and the War of Wrath 
ch24 = book[569589:594288]

# The Akallabeth
akallabeth = book[594292:657540]

# Of the Rings of Power and the Third Age
rings = book[657544:711150]


print(akallabeth)


# In[267]:


print(ch8)


# In[16]:


import pandas as pd
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

analyzer = SentimentIntensityAnalyzer()

sentiments_list = list()

for chapter in chapters:
    sentence_list = tokenize.sent_tokenize(chapter)
    sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        sentiments['compound'] += vs['compound']
        sentiments['neg'] += vs['neg']
        sentiments['neu'] += vs['neu']
        sentiments['pos'] += vs['pos']
            
    sentiments['compound'] = sentiments['compound'] / len(sentence_list)
    sentiments['neg'] = sentiments['neg'] / len(sentence_list)
    sentiments['neu'] = sentiments['neu'] / len(sentence_list)
    sentiments['pos'] = sentiments['pos'] / len(sentence_list)
    
    sentiments_list.append(sentiments)  # add this line

data4 = pd.DataFrame(sentiments_list)  # add this line
data4


# In[17]:


texts = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]



data = pd.DataFrame(data=[text for text in texts], columns=['Text'])

display(data.head(28))


# In[18]:


chapters = ['Ainulindalë','Valaquenta','Of the Beginning of Days','Of Aulë and Yavanna',
            'Of the Coming of the Elves and the Captivity of Melkor','Of Thingol and Melian',
            'Of Eldamar and the Princes of the Eldalië','Of Fëanor and the Unchaining of Melkor',
            'Of the Silmarils and the Unrest of the Noldor','Of the Darkening of Valinor','Of the Flight of the Noldor',
            'Of the Sindar','Of the Sun and Moon and the Hiding of Valinor','Of Men','Of the Return of the Noldor',
            'Of Beleriand and its Realms','Of the Noldor in Beleriand','Of Maeglin','Of the Coming of Men into the West',
            'Of the Ruin of Beleriand and the Fall of Fingolfin','Of Beren and Lúthien','Of the Fifth Battle: Nirnaeth Arnoediad',
            'Of Túrin Turambar','Of the Ruin of Doriath','Of Tuor and the Fall of Gondolin',
            'Of the Voyage of Eärendil and the War of Wrath','Akallabêth','Of the Rings of Power and the Third Age']

data2 = pd.DataFrame(chapters, columns=['Chapters'])

data3 = pd.merge(data, data2, left_index=True, right_index=True)

display(data3.head(28))


# In[19]:


# nltk
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd


chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

analyzer = SentimentIntensityAnalyzer()

sentiments_list = list()

for chapter in chapters:
    sentence_list = tokenize.sent_tokenize(chapter)
    sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        sentiments['compound'] += vs['compound']
        sentiments['neg'] += vs['neg']
        sentiments['neu'] += vs['neu']
        sentiments['pos'] += vs['pos']
            
    sentiments['compound'] = sentiments['compound'] / len(sentence_list)
    sentiments['neg'] = sentiments['neg'] / len(sentence_list)
    sentiments['neu'] = sentiments['neu'] / len(sentence_list)
    sentiments['pos'] = sentiments['pos'] / len(sentence_list)
    
    sentiments_list.append(sentiments)  # add this line

data4 = pd.DataFrame(sentiments_list)  # add this line
data4


# In[20]:


silmarillion_sentiments = pd.merge(data3, data4, left_index=True, right_index=True)

silmarillion_sentiments=silmarillion_sentiments.rename(columns={"compound": "Compound", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments


# In[21]:


from textblob import TextBlob

blob_book = TextBlob(book)
blob_ainulindale = TextBlob(ainulindale)
blob_valaquenta = TextBlob(valaquenta)
blob_ch1 = TextBlob(ch1)
blob_ch2 = TextBlob(ch2)
blob_ch3 = TextBlob(ch3)
blob_ch4 = TextBlob(ch4)
blob_ch5 = TextBlob(ch5)
blob_ch6 = TextBlob(ch6)
blob_ch7 = TextBlob(ch7)
blob_ch8 = TextBlob(ch8)
blob_ch9 = TextBlob(ch9)
blob_ch10 = TextBlob(ch10)
blob_ch11 = TextBlob(ch11)
blob_ch12 = TextBlob(ch12)
blob_ch13 = TextBlob(ch13)
blob_ch14 = TextBlob(ch14)
blob_ch15 = TextBlob(ch15)
blob_ch16 = TextBlob(ch16)
blob_ch17 = TextBlob(ch17)
blob_ch18 = TextBlob(ch18)
blob_ch19 = TextBlob(ch19)
blob_ch20 = TextBlob(ch20)
blob_ch21 = TextBlob(ch21)
blob_ch22 = TextBlob(ch22)
blob_ch23 = TextBlob(ch23)
blob_ch24 = TextBlob(ch24)
blob_akallabeth = TextBlob(akallabeth)
blob_rings = TextBlob(rings)


# In[22]:


blobs = [blob_ainulindale,blob_valaquenta,blob_ch1,blob_ch2,blob_ch3,blob_ch4,blob_ch5,blob_ch6,blob_ch7,
         blob_ch8,blob_ch9,blob_ch10,blob_ch11,blob_ch12,blob_ch13,blob_ch14,blob_ch15,blob_ch16,blob_ch17,
         blob_ch18,blob_ch19,blob_ch20,blob_ch21,blob_ch22,blob_ch23,blob_ch24,blob_akallabeth,blob_rings]

polarity_list = list()
subjectivity_list = list()

for blob in blobs:
    sentence_list1 = blob.sentences
    polarity = (0)
    subjectivity = (0)
        
    for sentence in sentence_list1:
        pl=sentence.sentiment.polarity
        polarity += pl
        sb=sentence.sentiment.subjectivity
        subjectivity += sb
            
    polarity = polarity / len(sentence_list1)
    subjectivity = subjectivity / len(sentence_list1)

    polarity_list.append(polarity)  
    subjectivity_list.append(subjectivity)

data5 = pd.DataFrame(polarity_list)  
data6 = pd.DataFrame(subjectivity_list)  
data7 = pd.merge(data5, data6, left_index=True, right_index=True)
data7=data7.rename(columns={'0_x': "Polarity", '0_y': 'Subjectivity'})

data7


# In[23]:


silmarillion_sentiments = pd.merge(silmarillion_sentiments, data7, left_index=True, right_index=True)

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments


# In[24]:


chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

wordlength_list = list()

for chapter in chapters:
    count = len(chapter)
        
    wordlength_list.append(count)  

data8 = pd.DataFrame(wordlength_list)  
data8=data8.rename(columns={0: 'Total_words'})
data8


# In[25]:


silmarillion_sentiments = pd.merge(silmarillion_sentiments, data8, left_index=True, right_index=True)

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments.head()


# In[35]:


def rating(silmarillion_sentiments):
    if silmarillion_sentiments['Compound'] > 0.05:
        return 'Positive'
    elif silmarillion_sentiments['Compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

silmarillion_sentiments['Rating'] = silmarillion_sentiments.apply(rating, axis=1)

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments.head(10)


# In[132]:


import io, re, nltk
from pathlib import Path
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist

chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

Noun_freq = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    words = [word for word in tokens if word not in stop]
    stop = set(stopwords.words('english'))
    tagged = pos_tag(words)
    nouns = [word for word, pos in tagged if (pos == 'NN')]
    Noun1 = (0)
            
    Noun1 = FreqDist(nouns).most_common(10)
    
    Noun_freq.append(Noun1)
    
Noun_freq


# In[141]:


# The Lexical Diversity represents the ratio of unique words used to the total number of words in the story.

chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

def lexical_density(text):
    return len(set(text)) / len(text)

Lexical_density = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    LexDen = (0)
            
    LexDen = lexical_density(tokens)

    Lexical_density.append(LexDen)  

LexDen = pd.DataFrame(Lexical_density) 
LexDen=LexDen.rename(columns={0: 'Lex_density'})
LexDen


# In[144]:


# The Lexical Diversity represents the ratio of unique words used to the total number of words in the story.

chapters = [ainulindale,valaquenta,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,ch17,
            ch18,ch19,ch20,ch21,ch22,ch23,ch24,akallabeth,rings]

def lexical_density(text):
    return len(set(text)) / len(text)

Lexical_density_norm = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    tokens = tokens[0:3440]
    LexDen1 = (0)
            
    LexDen1 = lexical_density(tokens)

    Lexical_density_norm.append(LexDen1)  

LexDen_norm = pd.DataFrame(Lexical_density_norm) 
LexDen_norm=LexDen_norm.rename(columns={0: 'Lex_density_norm'})
LexDen_norm


# In[139]:


silmarillion_sentiments = pd.merge(silmarillion_sentiments, LexDen, left_index=True, right_index=True)

silmarillion_sentiments = silmarillion_sentiments[['Chapters','Text','Total_words','Lex_density','Negative',
                                                  'Neutral','Positive','Compound','Rating','Polarity','Subjectivity']]

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments.head()


# In[147]:


silmarillion_sentiments = pd.merge(silmarillion_sentiments, LexDen_norm, left_index=True, right_index=True)

silmarillion_sentiments = silmarillion_sentiments[['Chapters','Text','Total_words','Lex_density','Lex_density_norm','Negative',
                                                  'Neutral','Positive','Compound','Rating','Polarity','Subjectivity']]

silmarillion_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments.head()


# ## NRC Sentiment

# In[186]:


silmarillion_sentiments = silmarillion_sentiments.replace('\n',' ', regex=True) 
silmarillion_sentiments.head()


# In[257]:


import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm_notebook as tqdm
from tqdm import trange


def text_emotion(df, column):
    '''
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns for each emotion
    '''

    new_df = df.copy()

    xlsx = pd.read_excel(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC-Sentiment-Emotion-Lexicons\NRC-Sentiment-Emotion-Lexicons\NRC-Emotion-Lexicon-v0.92\NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx')
    emolex_df = xlsx[['word', 'Positive_NRC','Negative_NRC','Anger', 'Anticipation', 'Disgust', 'Fear','Joy',
                      'Sadness', 'Surprise', 'Trust']]
    emotions = emolex_df.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")

    
    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in new_df.iterrows():
            pbar.update(1)
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_df[emolex_df.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df


# In[258]:


silmarillion_sentiments_final = text_emotion(silmarillion_sentiments, 'Text')


# In[259]:


silmarillion_sentiments_final.head()


# In[260]:


silmarillion_sentiments_final['word_count'] = silmarillion_sentiments_final['Text'].apply(tokenize.word_tokenize).apply(len)
silmarillion_sentiments_final


# In[240]:


silmarillion_sentiments_final.Anger = silmarillion_sentiments_final.Anger.astype('float64') 
silmarillion_sentiments_final.Anticipation = silmarillion_sentiments_final.Anticipation.astype('float64') 
silmarillion_sentiments_final.Disgust = silmarillion_sentiments_final.Disgust.astype('float64') 
silmarillion_sentiments_final.Fear = silmarillion_sentiments_final.Fear.astype('float64') 
silmarillion_sentiments_final.Joy = silmarillion_sentiments_final.Joy.astype('float64') 
silmarillion_sentiments_final.Negative = silmarillion_sentiments_final.Negative_NRC.astype('float64') 
silmarillion_sentiments_final.Positive = silmarillion_sentiments_final.Positive_NRC.astype('float64') 
silmarillion_sentiments_final.Sadness = silmarillion_sentiments_final.Sadness.astype('float64') 
silmarillion_sentiments_final.Surprise = silmarillion_sentiments_final.Surprise.astype('float64') 
silmarillion_sentiments_final.Trust = silmarillion_sentiments_final.Trust.astype('float64') 
silmarillion_sentiments_final.word_count = silmarillion_sentiments_final.word_count.astype('float64') 


# In[261]:


silmarillion_sentiments_final.dtypes


# In[262]:


emotions = ['Positive_NRC','Negative_NRC','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise',
                 'Trust']


# In[263]:


for emotion in emotions:
    silmarillion_sentiments_final[emotion]=silmarillion_sentiments_final[emotion]/silmarillion_sentiments_final['word_count']

silmarillion_sentiments_final.head()


# In[264]:


silmarillion_sentiments_final.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')


# In[269]:


def ratings(silmarillion_sentiments_final):
    if silmarillion_sentiments_final['Compound'] > 0.05:
        return 1
    elif silmarillion_sentiments_final['Compound'] < -0.05:
        return -1
    else:
        return 0

silmarillion_sentiments_final['Rating_num'] = silmarillion_sentiments_final.apply(ratings, axis=1)

silmarillion_sentiments_final.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')

silmarillion_sentiments_final.head(10)


# In[ ]:




