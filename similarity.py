import sys
import utils
import traceback
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from polyfuzz import PolyFuzz

###########################################################################################################################
# Author        : Tapas Mohanty  
# Modified      : 
# Reviewer      :
# Functionality : Fiding the similarity between two text using polyfuzz
# Comments      : https://github.com/MaartenGr/PolyFuzz
###########################################################################################################################

def similaritypolymain(pTrainData, pTestData, pAsg, pDesc, pDate, Nbest):
    try:
        pTrainData = pTrainData[pTrainData[pDesc].notna()]
        pTestData = pTestData[pTestData[pDesc].notna()]
        pTestData['Assignee_Group_Pred'], pTestData['Confidence_Level'] = 'Nan', float(0.0)
        pTrainDataDesc = pd.DataFrame(pTrainData[pDesc])
        pFeaList = []
        pFeaList = pTrainData['Features'].tolist() + pTestData['Features'].tolist()
        pFeaUnqList = list(set(pFeaList))   
        pMatchData, pData, pTestAppendDf,  = [], [], []
        pMatchesDf, pTestMatchData, pTestDf = pd.DataFrame(),pd.DataFrame(), pd.DataFrame()
        for i in range(len(pFeaUnqList)):
            ToData, FromData = pd.DataFrame(), pd.DataFrame()
            FromData = pTrainData.loc[pTrainData['Features'] == pFeaUnqList[i]]
            ToData = pTestData.loc[pTestData['Features'] == pFeaUnqList[i]]
            model = PolyFuzz("TF-IDF")
            pTestAppendDf.append(ToData)
            if len(ToData[pDesc].tolist()) and len(FromData[pDesc].tolist()) >= 1:
                model.match(list(ToData[pDesc].values), FromData[pDesc].unique().tolist(), nbest = int(Nbest))
                Matches = model.get_matches()
                pMatchData.append(Matches)
                pData.append(ToData)              
            
        pMatchesDf = pd.concat(pMatchData)
        pTestMatchData = pd.concat(pData) 
        pTestDf = pd.concat(pTestAppendDf)
        pMatchesDf.reset_index(inplace=True)
        del pMatchesDf['index']
        pTestMatchData.reset_index(inplace=True)
        del pTestMatchData['index']    
        pTestDf.reset_index(inplace=True)
        del pTestDf['index']        
        
        pTestConcatData = pd.concat([pTestMatchData,pMatchesDf], axis = 1)
        
        IntCol = ["To"]
        for i in range(1, int(Nbest)-1):
            IntCol.append("BestMatch" + "__" + str(i))
            pTestMatchData['Assignee_Group_Pred' + '__' + str(i)] = 'NaN'

        SimCol = ['Similarity']
        for k in range(1, int(Nbest) - 1):
            SimCol.append("Similarity" + "__" + str(k))
            pTestMatchData['Confidence_Level'+ '__' + str(k)] = 'NaN'
            
        for i in range(len(IntCol)):
            col = str(IntCol[i])
            if col != "To":
                pTestAppendFea = []
                for p in range(len(pFeaUnqList)):
                    pTrainFeaData, pTestFeaData = pd.DataFrame(), pd.DataFrame()
                    pTrainFeaData = pTrainData.loc[pTrainData['Features'] == pFeaUnqList[p]]
                    pTestFeaData = pTestConcatData.loc[pTestConcatData['Features'] == pFeaUnqList[p]]
                    pTestFeaData.reset_index(inplace=True)
                    del pTestFeaData['index'] 
                    if len(pTestFeaData) and len(pTrainFeaData)> 0: 
                        for j in range(len(pTestFeaData)):
                            if pMatchesDf[col][j] != None:
                                if len(pTrainFeaData[np.where(pTrainFeaData[pDesc] == pTestFeaData[IntCol[i]][j], True , False)][pAsg].values) != 0:
                                    pTestFeaData['Assignee_Group_Pred' + '__' + str(i-1)][j] = pTrainFeaData[np.where(pTrainFeaData[pDesc] == pTestFeaData[col][j], True , False)][pAsg].values[0]
            else:
                pTestAppendFea = []
                for p in range(len(pFeaUnqList)):
                    pTrainFeaData, pTestFeaData = pd.DataFrame(), pd.DataFrame()
                    pTrainFeaData = pTrainData.loc[pTrainData['Features'] == pFeaUnqList[p]]
                    pTestFeaData = pTestConcatData.loc[pTestConcatData['Features'] == pFeaUnqList[p]]
                    pTestFeaData.reset_index(inplace=True)
                    del pTestFeaData['index'] 
                    if len(pTestFeaData) and len(pTrainFeaData)> 0: 
                        for j in range(len(pTestFeaData)):
                            if pTestFeaData[col][j] != None:
                                if len(pTrainFeaData[np.where(pTrainFeaData[pDesc] == pTestFeaData[IntCol[i]][j], True , False)][pAsg].values) != 0:
                                    pTestFeaData['Assignee_Group_Pred'][j] = pTrainFeaData[np.where(pTrainFeaData[pDesc] == pTestFeaData[IntCol[i]][j], True , False)][pAsg].values[0]  
                                else:
                                    pTestFeaData['Assignee_Group_Pred'][j] =  None
                        pTestAppendFea.append(pTestFeaData)
        pTestFeaDf = pd.concat(pTestAppendFea)  
        pTestFeaDf.reset_index(inplace=True)
        del pTestFeaDf['index'] 
        
        pTestDf.loc[pTestDf['Number'].isin(pTestFeaDf['Number']), ['Confidence_Level', 'Assignee_Group_Pred']] = pTestFeaDf[['Similarity', 'Assignee_Group_Pred']].values
        
    except Exception as e:
        print('*** ERROR[004]: Error in similarity poly main function: ', sys.exc_info()[0],str(e))
        print(traceback.format_exc())
        return(-1)
        sys.exit(-1)
    return(0, pTestDf)    