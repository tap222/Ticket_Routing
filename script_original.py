# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:38:33 2019

@author: tamohant
"""

##Importing Libraries
import os
os.chdir('C:\\Users\\tamohant\\Desktop\\Ticket_Assignment\\new_data')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re, string


train = pd.read_excel('Training_Data_19_11_2019.xlsx',delimiter=',',encoding='latin-1')
test=pd.read_excel('testingdata.xlsx')

train['Description']=train['Short description'].astype('str')+' '+train['Description'].astype('str')
test['Description']=test['Short description'].astype('str')+' '+test['Description'].astype('str')

###Handling target values
train['Configuration_item']= train['Configuration item'].astype('category')
train = pd.concat([train,pd.get_dummies(train['Configuration_item'], prefix='Configuration_item')],axis=1)
train.drop(['Configuration_item'],axis=1, inplace=True)


#Label columns
label_cols = ['Configuration_item_powermax-bw',
       'Configuration_item_powermax-fi', 'Configuration_item_powermax-hr',
       'Configuration_item_powermax-mm', 'Configuration_item_powermax-pe1',
       'Configuration_item_powermax-pi1', 'Configuration_item_powermax-pp',
       'Configuration_item_powermax-ps', 'Configuration_item_powermax-sd']

##Filling the Missing vaues
COMMENT = 'Description'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


re_tok = re.compile(f'([{string.punctuation}“”¨_«»®´·º½¾¿¡§£₤‘’])')

 
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

#TF-IDF Vectorizer character level
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,5),
                      stop_words="english",
                      analyzer='char',
                      tokenizer=tokenize,
                      min_df=3, max_df=0.9, 
                      strip_accents='unicode',
                      use_idf=1,
                      smooth_idf=1, 
                      sublinear_tf=1)


trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

# trn_term_doc, test_term_doc
 
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc

#Logistic Regression
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4,solver='sag',dual=False,class_weight="balanced",n_jobs=-1)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))

#Prediction

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


#Submission File 
submid = pd.DataFrame({'Number': test["Number"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission['Confidence_level'] = submission[label_cols].max(axis=1)
submission['Pred_Configruation_item'] = submission[label_cols].idxmax(axis=1)
submission['Pred_Configruation_item'] = submission['Pred_Configruation_item'].str.replace('Configuration_item_', '')
submission = submission[['Number','Confidence_level','Pred_Configruation_item']]

# #Accuracy
# def accuracy(submission):
    # if submission['Pred_Configruation_item'].astype('category') == test['Configuration item'].astype('category'):
        # submission['Value'] = 1
    # else:
        # submission['Value'] = 0
        
    # acc = sum(submission['Value'].astype('int')) * 100
    
    # return acc
    
# accuracy(submission)  
submission.to_csv('Training_Data_19_11_2019_result.csv', index=False)
