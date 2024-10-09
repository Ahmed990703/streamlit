import pandas as pd
import streamlit as st
from pycaret.classification import *
from pycaret.datasets import get_data
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
import warnings
from sklearn.preprocessing import LabelEncoder

st.header('final project')
le=LabelEncoder()
    
file=st.file_uploader('Upload file',type=["csv"])
if file is not None:
 df=pd.read_csv(file)
 st.write(df)
 
 
 numeric_columns = df.select_dtypes(include='number').columns
 categorical_columns = df.select_dtypes(exclude='number').columns
 x=st.selectbox('choose column to be your label',df.columns)
 if pd.api.types.is_numeric_dtype(df[x].dtype):
    st.write('this a Regression task')
    warnings.filterwarnings("ignore")
    if df.isnull().any:
      st.text('Your data has null values')
      handle=st.selectbox('choose how to handle your null values in numerical columns',['fill with mean value','fill with median value','fill value by set value by your self'])
      if handle=='fill with mean value':
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))
        st.write(df)
      if handle=='fill with median value':
       df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))
       st.write(df)
      if handle=='fill value by set value by your self':
        value2=st.number_input('Set your value')
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(value2))
      handle2=st.selectbox('choose how to handle your null values in categorical columns',['fill with most frequent value','fill value by set value by your self'])
      if handle2=='fill with most frequent value':
       df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))
       st.write(df)
      if handle2=='fill value by set value by your self':
        value2=st.text_input('Set your value')
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(value2))
        st.write(df)
    dropped=st.multiselect("choose columns to be dropped",df.columns)
    df=df.drop(dropped,axis=1)
    st.header('data after dropped unnecessary columns')
    st.write(df)
    tab1,tab2=st.tabs(['Scatter plot','histogram'])
    with tab1:
      numerical_cols=df.select_dtypes(include=np.number).columns.to_list()
      x_cols=st.multiselect('choose colums to be shown',numerical_cols,numerical_cols,key="scatter_x_cols")
      y_cols=st.multiselect('choose colums to be shown',numerical_cols,numerical_cols,key="scatter_y_cols")
      fig_scatter=px.scatter(x_cols,y_cols)
      st.plotly_chart(fig_scatter)
    with tab2:
      hist_feature=st.multiselect('choose feature to be show in histogram',numerical_cols,key="unique_key_for_y_cols")
      fig_hist=px.histogram(df,x=hist_feature)
      st.plotly_chart(fig_hist)
    for column in df.select_dtypes(include=['object']).columns:
      df[column]= le.fit_transform(df[column])
    st.text('data after encoding')
    st.write(df)
    from pycaret.regression import *
    #st.text(df[x].value_counts)
    reg=setup(data=df,target=f"{x}")
    best=compare_models()
    st.header('The best model for this task will be :\n')
    st.title(f'{best}')
 elif pd.api.types.is_object_dtype(df[x].dtype):
  st.write('this a classification task')
  warnings.filterwarnings("ignore")
  if df.isnull().any:
      st.text('Your data has null values')
      handle=st.selectbox('choose how to handle your null values in numerical columns',['fill with mean value','fill with median value','fill value by set value by your self'])
      if handle=='fill with mean value':
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))
        st.write(df)
      elif handle=='fill with median value':
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()))
        st.write(df)
      elif handle=='fill value by set value by your self':
        value=st.number_input('Set your value')
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(value))
      handle2=st.selectbox('choose how to handle your null values in categorical columns',['fill with most frequent value','fill value by set value by your self'])
      if handle2=='fill with most frequent value':
       df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(x.mode()[0]))
       st.write(df)
      elif handle=='fill value by set value by your self':
        value2=st.text_input('Set your value')
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.fillna(value2))
        st.write(df)
  dropped=st.multiselect("choose columns to be dropped",df.columns)
  df=df.drop(dropped,axis=1)
  st.header('data after dropped unnecessary columns')
  st.write(df)
  tab1,tab2=st.tabs(['Scatter plot','histogram'])
  with tab1:
    numerical_cols=df.select_dtypes(include=np.number).columns.to_list()
    x_cols=st.multiselect('choose colums to be shown',numerical_cols,key="scatter_x_cols")
    y_cols=st.multiselect('choose colums to be shown',numerical_cols,key="scatter_y_cols")
    fig_scatter=px.scatter(x_cols,y_cols)
    st.plotly_chart(fig_scatter)
  with tab2:
    hist_feature=st.multiselect('choose feature to be show in histogram',numerical_cols, key="unique_key_for_y_cols")
    fig_hist=px.histogram(df,x=hist_feature)
    st.plotly_chart(fig_hist)
  for column in df.select_dtypes(include=['object']).columns:
     df[column]= le.fit_transform(df[column])
  st.text('data frame after encoding')
  st.write(df)
  from pycaret.classification import *
  reg=setup(data=df,target=f"{x}")
  best=compare_models()
  st.header('The best model for this task will be :\n')
  st.title(f'{best}')


      
 
