import pickle
import pandas as pd

subdataset_filepath = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1\piaac_norway_subdataset1.pkl'
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

df = loaded_data['dataframe']
print("Unique PAREDC2 values:", df['PAREDC2'].unique())
print("Unique A2_Q03a_T values:", df['A2_Q03a_T'].unique())
print("Unique IMPARC2 values:", df['IMPARC2'].unique())
print("Unique ED_GROUP values:", df['ED_GROUP'].unique())
print("Unique GENDER_R values:", df['GENDER_R'].unique())
