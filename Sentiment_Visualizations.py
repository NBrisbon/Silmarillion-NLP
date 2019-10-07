#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[215]:


silmarillion_sentiments=pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\silmarillion_sentiments.csv')
silmarillion_sentiments=silmarillion_sentiments.drop(columns=['Unnamed: 0'])
silmarillion_sentiments.head()


# In[216]:


silmarillion_sentiments.describe()


# In[198]:


import matplotlib.patches as mpatches

kwargs = dict(alpha=0.6)

patch1 = mpatches.Patch(label='Normalized: 3440 tokens', **kwargs)
patch2 = mpatches.Patch(color='b', label='Full token count', **kwargs)
all_handles = (patch1, patch2)

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_alpha(0.7)
ax.barh(silmarillion_sentiments['Chapters'], silmarillion_sentiments['Lex_density_norm'],alpha=.5)
ax.barh(silmarillion_sentiments['Chapters'], silmarillion_sentiments['Lex_density'],color='b',alpha=.7)
ax.set_title("Lexical Density by Chapter in The Silmarillion",fontsize=33)
ax.set_xlabel("Lexical Density Score", fontsize=27)
ax.set_ylabel("Chapters", fontsize=27)
#ax.set_xticklabels([-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25],fontsize=20)
ax.set_yticklabels(silmarillion_sentiments.Chapters, rotation=0, fontsize=22)
ax.legend(handles=all_handles,loc='lower right', fontsize=22)
ax.tick_params(axis='x', which='major', labelsize=18)
ax.invert_yaxis()
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\Lexical_Diversity.png',bbox_inches='tight')


# In[199]:


import matplotlib.patches as mpatches

kwargs = dict(alpha=0.5)

patch1 = mpatches.Patch(color='g', label='Positive Rating (>0.05)', **kwargs)
patch2 = mpatches.Patch(color='r', label='Negative Rating (<-0.05)', **kwargs)
patch3 = mpatches.Patch(color='orange', label='Neutral Rating (-0.05 - 0.05)', **kwargs)
all_handles = (patch1, patch2, patch3)

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_alpha(0.5)
ax.barh(silmarillion_sentiments['Chapters'], silmarillion_sentiments['Compound'],
        color=silmarillion_sentiments.Rating.map({'Positive': 'g', 'Negative': 'r', 'Neutral': 'orange'}),
        alpha=.5)
ax.set_title("Sentiment Compound Scores in The Silmarillion (VADER)",fontsize=33)
ax.set_xlabel("Compound Score (Range= -1.0 - 1.0)", fontsize=27)
ax.set_ylabel("Chapters", fontsize=27)
#ax.set_xticklabels([-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25],fontsize=20)
ax.set_yticklabels(silmarillion_sentiments.Chapters, rotation=0, fontsize=22)
ax.legend(handles=all_handles,loc='lower right', fontsize=20)
ax.tick_params(axis='x', which='major', labelsize=18)
ax.invert_yaxis()
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\VADER_compound.png',bbox_inches='tight')


# In[148]:


print ('--- Sentiment Scores for The Silmarillion Averaged Across Chapters ---')
print ('\n')
print ('-- TEXT BLOB --')
print ('Polarity: {0:.3f}'.format(silmarillion_sentiments['Polarity'].mean()))
print ('Subjectivity: {0:.3f}'.format(silmarillion_sentiments['Subjectivity'].mean()))
print ('\n')
print ('-- VADER --')
print ('Positive: {0:.3f}'.format(silmarillion_sentiments['Positive'].mean()))
print ('Negative: {0:.3f}'.format(silmarillion_sentiments['Negative'].mean()))
print ('Neutral: {0:.3f}'.format(silmarillion_sentiments['Neutral'].mean()))
print ('Compound: {0:.3f}'.format(silmarillion_sentiments['Compound'].mean()))
print ('\n')
print ('-- NRC --')
print ('Positive: {0:.3f}'.format(silmarillion_sentiments['Positive_NRC'].mean()))
print ('Joy: {0:.3f}'.format(silmarillion_sentiments['Joy'].mean()))
print ('Anticipation: {0:.3f}'.format(silmarillion_sentiments['Anticipation'].mean()))
print ('Surprise: {0:.3f}'.format(silmarillion_sentiments['Surprise'].mean()))
print ('Trust: {0:.3f}'.format(silmarillion_sentiments['Trust'].mean()))
print ('Negative: {0:.3f}'.format(silmarillion_sentiments['Negative_NRC'].mean()))
print ('Anger: {0:.3f}'.format(silmarillion_sentiments['Anger'].mean()))
print ('Fear: {0:.3f}'.format(silmarillion_sentiments['Fear'].mean()))
print ('Disgust: {0:.3f}'.format(silmarillion_sentiments['Disgust'].mean()))
print ('Sadness: {0:.3f}'.format(silmarillion_sentiments['Sadness'].mean()))


# In[200]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Compound',figsize=(25,15), color='blue',
                             alpha=.35, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Negative',figsize=(25,15), color='red',
                             alpha=.35, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Neutral',figsize=(25,15), color='purple',
                             alpha=.35, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Positive',figsize=(25,15), color='green',
                             alpha=.35, linewidth=7, ax=ax)
plt.axhline(y=0, xmin=0, xmax=1, alpha=.5, color='orange', linestyle='--', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Silmarillion (VADER)', fontsize=40)
plt.xlim(-1,28)
plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\VADER_full.png',bbox_inches='tight')


# In[201]:


ax1 = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Compound',figsize=(25,10), color='blue',
                             alpha=.35, linewidth=7, ax=ax1)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Negative',figsize=(25,10), color='red',
                             alpha=.35, linewidth=7, ax=ax1)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Positive',figsize=(25,10), color='green',
                             alpha=.35, linewidth=7, ax=ax1)
plt.axhline(y=0, xmin=0, xmax=1, alpha=.5, color='orange', linestyle='--', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Silmarillion (VADER: without "Neutral")', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.5,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks([-0.15,-0.10,-0.05,0,0.05,0.10,0.15,0.20,0.25],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\VADER_zoom.png',bbox_inches='tight')


# In[202]:


ax2 = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Polarity',figsize=(25,10), color='blue',
                             alpha=.35,linewidth=7, ax=ax2)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Subjectivity',figsize=(25,10), color='orange', 
                             alpha=.35, linewidth=7, ax=ax2)
plt.legend(loc='best', fontsize=25)
plt.title('Polarity/Subjectivity of The Silmarillion (TextBlob)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.5,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\TextBlob_polarity.png',bbox_inches='tight')


# In[203]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Positive_NRC',figsize=(25,15), color='green',
                             alpha=.35, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Negative_NRC',figsize=(25,15), color='red',
                             alpha=.35, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_PosNeg.png',bbox_inches='tight')


# In[204]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Anger',figsize=(25,15), color='red',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Disgust',figsize=(25,15), color='purple',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Fear',figsize=(25,15), color='maroon',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Sadness',figsize=(25,15), color='black',
                             alpha=.45, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment (Negative) of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_Neg.png',bbox_inches='tight')


# In[205]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Anticipation',figsize=(25,15), color='blue',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Joy',figsize=(25,15), color='orange',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Surprise',figsize=(25,15), color='lightblue',
                             alpha=.45, linewidth=7, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Trust',figsize=(25,15), color='green',
                             alpha=.45, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment (Positive) of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_Pos.png',bbox_inches='tight')


# In[206]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Sadness',figsize=(25,15), color='blue',
                             alpha=.35, linewidth=8, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Joy',figsize=(25,15), color='yellow',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Joy/Sadness Sentiments of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_JoySad.png',bbox_inches='tight')


# In[207]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Anticipation',figsize=(25,15), color='orange',
                             alpha=.35, linewidth=8, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Surprise',figsize=(25,15), color='teal',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Surprise/Anticipation Sentiments of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_SurAnt.png',bbox_inches='tight')


# In[208]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Trust',figsize=(25,15), color='green',
                             alpha=.35, linewidth=8, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Disgust',figsize=(25,15), color='purple',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Trust/Disgust Sentiments of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_TrustDis.png',bbox_inches='tight')


# In[209]:


ax = plt.gca()
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Anger',figsize=(25,15), color='red',
                             alpha=.35, linewidth=8, ax=ax)
silmarillion_sentiments.plot(kind='line',x='Chapters', y='Fear',figsize=(25,15), color='forestgreen',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Anger/Fear Sentiments of The Silmarillion (NRC)', fontsize=40)
plt.xlim(-1,28)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(28), silmarillion_sentiments.Chapters[0:28], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC_AngerFear.png',bbox_inches='tight')

