#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self,X,y=None,copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),columns = self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled],axis=1)[init_col_order]
    
    # creating special class that we'll use to predict new data
class absenteeism_model():
          
    def __init__(self,model_file,scaler_file):
        # reading the saved model and scaler files
         with open('model','rb') as model_file, open('scaler','rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.data = None
    
    # taking the data file and preprocessing it
    def load_and_clean_data(self,data_file):
            
        df = pd.read_csv(data_file,delimiter=",")
        self.df_with_predictions = df.copy() # saving for later use
        df = df.drop(['ID'],axis = 1)
            
        # preserving the code 
        df['Absenteeism Time in Hours']= 'Nan'
            
        # creating a seperate dataframe, containing dummy values for ALL available reasons
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
            
        # splitting reason columns into 4 types
        reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
            
        # dropping 'Reason for Absense' column to avoid multicollinearity
        df.drop(['Reason for Absence'],axis =1)
            
        # concatenating the 4 reason type columns
        df=pd.concat([df,reason_type_1,reason_type_2,reason_type_3,reason_type_4], axis = 1)

        # assigning proper names to the 4 reason type columns
        column_names=['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                      'Daily Work Load Average', 'Body Mass Index', 'Education',
                      'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1','Reason_2','Reason_3','Reason_4']
        df.columns = column_names
           
        # reordering the columns
        column_names_order=['Reason_1','Reason_2','Reason_3','Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education',
                            'Children', 'Pets', 'Absenteeism Time in Hours']
        df=df[column_names_order]
            
        # converting 'Date' column into datetime
        df['Date']=pd.to_datetime(df['Date'], format='%d/%m/%Y')
            
        # creating a list with month values retrieved from 'Date' column
        list_months =[]
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
                
        # inserting the values in a new column in df, 'Month Value'
        df['Month Value'] = list_months
            
        # creating a new feature called 'Day of the Week'
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
            
        # drop the 'Date' column from df
        df = df.drop(['Date'], axis =1)
            
        # reordering the columns in df
        col_updated=['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value',
                 'Day of the Week','Transportation Expense', 'Distance to Work', 'Age',
                 'Daily Work Load Average', 'Body Mass Index', 'Education',
                 'Children', 'Pets', 'Absenteeism Time in Hours']

        df = df[col_updated]
            
        # map 'Education' variables; the result is dummy
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
            
        # replace the NaN values
        df = df.fillna(value = 0)
            
        # dropping the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
            
        # dropping the variables we decide we don't need
        df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'], axis = 1)
            
        # 'preprocessed data'
        self.preprocessed_data = df.copy()
            
        # for next functions-
        self.data = self.scaler.transform(df)
            
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if(self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
            
    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if(self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
            
    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if(self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

