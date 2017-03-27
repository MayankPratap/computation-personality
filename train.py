
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
df=pd.read_csv('mypersonality_final.csv',encoding='ISO-8859-1')

# In[2]:


statusUpdates={}  # Contain all text of a single person from all status updates
pTrait={}         # Personality traits of a single person
cnt=0
for index,row in df.iterrows():
    #cnt=cnt+1
    #if cnt==200:
    #    break    
    try:
        statusUpdates[row['#AUTHID']]+=row['STATUS']
    except KeyError:
        statusUpdates[row['#AUTHID']]=row['STATUS']
    #print(index)
    #print(row['#AUTHID']+" "+row['cEXT']+" "+row['cNEU'])
    
    if row['#AUTHID'] not in pTrait:
        pTrait[row['#AUTHID']]=row['cEXT']

print("Total number of data points: "+str(len(statusUpdates)))
#print(statusUpdates["b7b7764cfa1c523e4e93ab2a79a946c4"])
"""   
cnt=0
for key in statusUpdates:
    cnt=cnt+1
    if cnt==3:
        break
    print(key+" "+statusUpdates[key])
"""
"""
cnt=0
}for key in pTrait:
    cnt=cnt+1
    #if cnt==30:
     #   break
    print(key+" "+str(pTrait[key]))
    
# Now creating labels_train and labels_test 

"""

labels_train=[]
labels_test=[]
features_train=[]   # List Of Feature Vectors, one feature vector for one data point(userid)
features_test=[]

for key in pTrait:
    if pTrait[key]=='y':
        labels_train.append(1)
    else:
        labels_train.append(0)

#for i in labels_train:
 #   print(labels_train[i])
    
l2=int(0.67*len(labels_train))

for i in range(l2+1,len(labels_train)):
    labels_test.append(labels_train[i])

labels_train=labels_train[:l2+1]
    
print(len(labels_train))
print(len(labels_test))

# In[3]:

from wordsegment import segment
from nltk import word_tokenize,pos_tag
import numpy as np
import enchant
import nltk 
import string
import csv

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re


# In[4]:

words_custom=[]
sentence_list=[]
pos_tags=[]
words_posscore={}
words_neg={}
words_pos={}
func_words=[]

def init():
    global words_posscore
    global words_neg
    global words_pos
    global func_words

    words_posscore={}

    with open('words_posscore.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            #print(type(row))
            #print(row)
            words_posscore[row[0]]=float(row[1])
    words_neg={}
    with open('negative_words.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            words_neg[row[0]]=1

    words_pos={}

    with open('positive_words.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            words_pos[row[0]]=1
    
    func_words=[]
    with open('func_words.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            func_words.append(row[0])


def get_pos_score(text):
    words=words_custom
    count=0
    score=0.0

    for word in words:
        if word in words_posscore:
            count+=1
            score+=words_posscore[word]

    return score
    
def get_neg_count(text):
    words=words_custom
    count=0
    
    for word in words:
        if word in words_neg:
            count+=1
    return count

def get_pos_count(text):
    words=words_custom
    count=0

    for word in words:
        if word in words_pos:
            count+=1

    return count

def get_func_count(text):
    words=words_custom
    count=0

    for word in words:
        if word in func_words:
            count+=1

        elif any(word in x for x in func_words):
            count+=1

    return count

from enchant.tokenize import get_tokenizer,HTMLChunker    

def custom_word_tokenize(text):
    tokenizer=get_tokenizer("en_US")
    words=[]

    for w in tokenizer(text):
        words.append(w[0])

    return words

def pref_for_longer_words(text):
    words=words_custom
    count_6=0
    count_7=0
    count_8=0
    count_9=0
    
    for word in words:
        length=len(word)
        
        if length>=6:
            count_6+=1
        if length>=7:
            count_7+=1

        if length>=8:
            count_8+=1

        if length>=9:
            count_9+=1
        
    return count_6,count_7,count_8,count_9


def removeSmileys(text):
    
    should_match=[
        
       ":'(",
       ":)",
       ":D",
       ":(",  # frown
       ":P",
       "O:)",
       "3:)",
       ";)",
       ":O"
       "-_-",
       ">:O",
       ":*",
       "<3",
       "^_^",
       "8-)",
       "8|",
       ">:(",
       ":/",
       "(y)",
       ":poop:",    
        
    ]
    big_regex=re.compile('|'.join(map(re.escape,should_match)))
    text=big_regex.sub(" ",text)
    
    text=' '.join(text.split())
    
    return text
    
def removeNumbers(text):
    text=re.sub(r'\d+','',text)
    return text    

punctuations=['.','"','{','(','-','!','?',':']

def count_punctuations(text):
    
    text=removeSmileys(text)
    text=removeNumbers(text)
    num_sentences=len(sentence_list)
    dots=text.count('.')

    punct_count=0

    for punctuation in punctuations:
        if punctuation == '.':
            punct_count+=num_sentences

        elif punctuation=='"':
            punct_count+=int(text.count(punctuation)/2)
        else:
            punct_count+=int(text.count(punctuation))


    return punct_count

# Richness in vocabulary 

from nltk.stem.porter import PorterStemmer
from itertools import groupby

def vocabulary_richness(text):
    # Yule's I measure ( the inverse of Yule's K Measure)
    # higher number is higher diversity - richer vocabulary
    d={}
    
    stemmer=PorterStemmer()
    words=words_custom

    for w in words:
        w=stemmer.stem(w).lower()

        # nltk 3.2.2 has a bug which caused string out Of index error
        # so I downgraded to nltk 3.2.1  :)  :) 

        try:
            d[w]+=1
        except KeyError:
            d[w]=1

    M1=float(len(d))
    M2=sum([len(list(g))*(freq**2) for freq,g in groupby(sorted(d.values()))])

    # M1 is the number of all words forms a text consists of
    # M2 is the sum of products of each observed frequency(^2) and number of word types observed with that frequency
    
    try:
        return (M1*M1)/(M2-M1) # Measure of Yule's I # Larger Yule's I the larger the diversity of vocabulary 

    except ZeroDivisionError:
        return 0
    
"""Words per sentence"""

def getWordsPerSentence(text):
    word_count=0
    word_sent_list=[]
    number_sentences=len(sentence_list)
    number_words=len(words_custom)
    return (number_words/number_sentences)

""" 
  I-Measure=(Wrong-typed Words freq. + Interjections freq. + Emoticon freq. )*100
"""

def getIMeasure(text):
    emotion_count=getEmotionCount(text)
    interjection_count=getInterjectionCount(text)
    wrong_word_count=getWrongWordCount(text)
    informality_measure=(interjection_count+wrong_word_count+emotion_count)*100
    return informality_measure,interjection_count,wrong_word_count

def getInterjectionCount(text):
    tuples=pos_tags
    intCount=0

    for t in tuples:
        if t[1]=="UH":
            intCount=intCount+1
    return intCount

def getWrongWordCount(text):
    US_spell_dict=enchant.Dict("en_US")
    British_spell_dict=enchant.Dict("en_GB")
    words_list=words_custom
    wrongCount=0

    for word in words_list:
        if US_spell_dict.check(word)==False and British_spell_dict.check(word)==False:
            wrongCount=wrongCount+1

    return wrongCount

def test_match(s,essay):
    return essay.count(s)

def getEmotionCount(text):
    should_match=[
        
        ":'(",
       ":)",
       ":D",
       ":(",  # frown
       ":P",
       "O:)",
       "3:)",
       ";)",
       ":O"
       "-_-",
       ">:O",
       ":*",
       "<3",
       "^_^",
       "8-)",
       "8|",
       ">:(",
       ":/",
       "(y)",
       ":poop:",    
        
    ]
    
    
    count=0
    
    for x in should_match:
        count=count+test_match(x,text)
    
    return count

# TypeBYTokenRation
def gettr_ratio(text):
    words=words_custom
    return len(set(words))*1.0/len(words)

"""
  F-measure=(noun freq+ adjective freq + preposition freq + article freq - pronoun freq - verb freq - adverb freq - interjection freq + 100)/2
"""
def getcounts(text):
    tuples=pos_tags
    noun_count=0 
    adj_count=0 #JJ
    preposition_count=0 #IN
    article_count=0 # a, an, the
    pronoun_count=0 # PR 
    verb_count=0 #VB
    adverb_count=0 #RB
    interjection_count=0 
    
    for t in tuples:
        if 'NN' in t[1]:
            noun_count+=1
        elif 'JJ' in t[1]:
            adj_count+=1
        elif 'IN' in t[1]:
            preposition_count+=1
        elif 'PR' in t[1]:
            pronoun_count+=1
        elif 'VB' in t[1]:
            verb_count+=1
        elif 'RB' in t[1]:
            adverb_count+=1
        elif 'UH' in t[1]:
            interjection_count+=1
        elif t[0]=="a" or t[0]=="A" or t[0]=="an" or t[0]=="An" or t[0]=="the" or t[0]=="The" or t[0]=="THE":
            article_count+=1
        
    result=[]
    result.append(noun_count)
    result.append(adj_count)
    result.append(preposition_count)
    result.append(pronoun_count)
    result.append(verb_count)
    result.append(adverb_count)
    result.append(interjection_count)
    result.append(article_count)
    
    return result

def getFMeasure(text):
    words=words_custom
    noun_count=0  # NN
    for word in words:
        if word=="*PROPNAME*":
            noun_count+=1
            
    adj_count=0   #JJ
    preposition_count=0 #IN
    article_count=0 # a, an, the
    pronoun_count=0 # PR
    verb_count=0 #VB
    adverb_count=0 # RB
    interjection_count=0 
    
    result=[]
    result=getcounts(text)

    noun_count+=result[0]
    adj_count+=result[1]
    preposition_count+=result[2]
    pronoun_count+=result[3]
    verb_count+=result[4]
    adverb_count+=result[5]
    interjection_count+=result[6]
    article_count+=result[7]
    
    
    FMeasure=(noun_count+adj_count+preposition_count+article_count-pronoun_count-verb_count-adverb_count-interjection_count+100)/2

    return FMeasure

tentative_list=['may','might','maybe','mightbe','can','could' ,'perhaps','conceivably','imaginably','reasonably','perchance','feasibly','credible','obtainable','probably']

def gettentMeasure(text):
    words=words_custom
    c=0
    for word in words:
        if word in tentative_list:
            c+=1

    return c

def tense_count(text):
    pos_tagged_text=pos_tags
    VB=[]    # Verb in base form
    VBD=[]   # Verb in past tense
    VBG=[]   # Verb in present participle
    VBP=[]   # Verb in non-3rd person singular present
    VBN=[]   # Verb past participle
    VBZ=[]   # Verb in 3rd person singular present
    Others=[]
    
    for (word,tag) in pos_tagged_text:
        if tag=="VB":
            VB.append(word)
        elif tag=="VBD":
            VBD.append(word)
        elif tag=="VBG":
            VBG.append(word)
        elif tag=="VBN":
            VBN.append(word)
        elif tag=="VBP":
            VBP.append(word)
        elif tag=="VBZ":
            VBZ.append(word)
        elif tag.find("VB")!=-1:
            Others.append(word)
        
    number_past_tense=len(VBD)+len(VBP)
    number_present_tense=len(VBG)
    
    return number_past_tense,number_present_tense

def get_all_features(text):
    global words_custom
    global sentence_list
    global pos_tags

    words_custom=custom_word_tokenize(text)
    sentence_list=nltk.sent_tokenize(text)
    pos_tags=nltk.pos_tag(words_custom)

    number_words=len(words_custom)
    number_sentences=len(sentence_list)

    features=[]
    # Type By Token Ratio
    type_by_token_ratio=gettr_ratio(text)
    features.append(type_by_token_ratio)

    # F-Measure
    f_measure=getFMeasure(text)
    features.append(f_measure)
    
    # I-Measure
    i_measure,interjection_count,wrong_word_count=getIMeasure(text)
    features.append(i_measure)
    
    # features.append(interjection_count/number_words)
    features.append(wrong_word_count/number_words)

    # Words per sentence
    words_per_sentence=getWordsPerSentence(text)
    features.append(words_per_sentence)

    # Preference to longer words
    count_6,count_7,count_8,count_9=pref_for_longer_words(text)
    features.append(count_6/number_words)
    features.append(count_7/number_words)
    features.append(count_8/number_words)
    features.append(count_9/number_words)

    # Tentativity 
    # tentativity=gettentMeasure(text)
    # features.append(tentativity/number_words)
    # Tense Count
    number_past_tense,number_present_tense=tense_count(text)
    features.append(number_past_tense/number_words)
    features.append(number_present_tense/number_words)

    # Punctuation Count
    number_punctuations=count_punctuations(text)
    features.append(number_punctuations/number_sentences)

    # Vocabulary richness
    vocabulary_richness_measure=vocabulary_richness(text)
    features.append(vocabulary_richness_measure)

    # Positive score
    positive_score=get_pos_score(text)
    
    features.append(positive_score)

    # Positive and negative words
    # positive_words=get_pos_count(text)
    # negative_words=get_neg_count(text)
    # features.append(positive_words)
    # features.append(negative_words)

    # Functional words
    # functional_words=get_func_count(text)
    # features.append(functional_words)

    # emoticon_count=findEmoticonCount(essay)
    # features.append(emoticon_count)

    return features
    
init()
cnt=0
for key in statusUpdates:
    #cnt=cnt+1
    #if cnt==200:
    #    break

    del words_custom[:]
    del sentence_list[:]
    del pos_tags[:]
    #print("Hello")
    #print(cnt+1)
    cnt=cnt+1
    text=statusUpdates[key]
    
    #statusUpdates[row['#AUTHID']]=row['STATUS']
    #print(index)
    #print(row['#AUTHID']+" "+row['cEXT']+" "+row['cNEU'])
    features_train.append(get_all_features(text))

"""
for i in features_train:
    print(i)
    print("\n\n\n")
"""    

l2=int(0.67*len(features_train))

for i in range(l2+1,len(features_train)):
    features_test.append(features_train[i])

features_train=features_train[:l2+1]

#print(len(features_train))
#print(len(features_test))

from sklearn import svm
from sklearn.metrics import accuracy_score
clf=svm.SVC()

clf.fit(features_train,labels_train)
pred=clf.predict(features_test)

score=accuracy_score(labels_test,pred)

print(score)




