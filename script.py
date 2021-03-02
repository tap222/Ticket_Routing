import pandas as pd
import numpy as np
from scipy.sparse import hstack 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train = pd.read_excel('TrainData.xlsx')
test = pd.read_excel('TestData.xlsx', sheet_name = 'Data')

###Handling target values
train['Ticket_Description']= train['Ticket_Description'].astype('str')
train['Assignment_Group'] = train['Assignment_Group'].astype('category')
train = pd.concat([train,pd.get_dummies(train['Assignment_Group'])],axis=1)
label_cols =  train['Assignment_Group'].cat.categories.tolist()
train.drop(['Assignment_Group'],axis=1, inplace=True)

def features(pData, pDate):
    try:
        #Creating Features with date
        pData[pDate] = pd.to_datetime(pData[pDate],format="%Y/%m/%d")
        pData['date_feat'] = pData[pDate]
        pData['dayofmonth'] = pData.date_feat.dt.day.astype('str')
        pData['dayofyear'] = pData.date_feat.dt.dayofyear.astype('str')
        pData['year'] = pData.date_feat.dt.year.astype('str')
        pData['month'] = pData.date_feat.dt.month.astype('str')
        pData['dayofweek_name'] = pData.date_feat.dt.day_name().astype('str') 
        pData['weekofyear'] = pData.date_feat.dt.weekofyear.astype('str')
        #pData['FeaSum'] = pData['dayofmonth'] + pData['dayofyear']  + pData['year'] + pData['month']
        
    except Exception as e:
        raise(e)
        print(traceback.format_exc())
        print('*** Error[007]: in count prediction file ocurred for ocurred in creating features :', e)
            
    return pData
    
#Label Encoder Features 
# trainfea = features(pData = train, pDate = 'Created')
# testfea = features(pData = test, pDate = 'Created')
# encode = LabelEncoder()
# Trainlabenc= trainfea[['FeaSum']].apply(encode.fit_transform)
# Testlabenc = testfea[['FeaSum']].apply(encode.fit_transform)

#Handling Other Features
train = features(pData = train, pDate = 'Created')
test = features(pData = test, pDate = 'Created')
encoder = OneHotEncoder(categories = "auto", handle_unknown ='ignore')
Train_encoded = encoder.fit_transform(train[['Category', 'Subcategory','Business_Area', 'dayofmonth', 'dayofyear', 'year', 'month', 'dayofweek_name', 'weekofyear']])
Test_encoded = encoder.transform(test[['Category', 'Subcategory','Business_Area', 'dayofmonth', 'dayofyear', 'year', 'month', 'dayofweek_name', 'weekofyear']])
train.drop(['Category', 'Subcategory'],axis=1, inplace=True)

##Filling the Missing vaues
COMMENT = 'Ticket_Description'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
    
# x = trn_term_doc
x = hstack([trn_term_doc, Train_encoded]).tocsr()
# x = hstack([trn_term_doc, Train_encoded, Trainlabenc]).tocsr()

#test_x = test_term_doc
test_x = hstack([test_term_doc, Test_encoded]).tocsr()
# test_x = hstack([test_term_doc, Test_encoded, Testlabenc]).tocsr()

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False,class_weight="balanced")
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

submid = pd.DataFrame({'Number': test["Number"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission['Confidence_level'] = submission[label_cols].max(axis=1)
submission['Assignee_Group_Pred'] = submission[label_cols].idxmax(axis=1)
#submission[['Level1','Level2']] = submission.Intent.str.split("__",expand=True,)
submission = submission[['Number','Confidence_level','Assignee_Group_Pred']]
submission.to_excel('TestResult_oheFeastrexp.xlsx', index=False)