from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pickle


df=pd.read_csv("C:\\Users\\RAGUL M\\Documents\\Web development\\Bio-ML\\1520792014405.csv")

km=KMeans(n_clusters=3)
Y_predicted=km.fit_predict(df[['heart_pulse','temperature','oxygen_saturation']])

pickle.dump(km,open('MLmodel.pkl','wb'))
MLmodel=pickle.load(open('MLmodel.pkl','rb'))