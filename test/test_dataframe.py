# Creating a dataframe and accessing the data
import pandas as pd
import sys
df = pd.DataFrame({
    'name': ['Mike', 'Alice', 'Bob'],
    'age': [30, 45, 80],
    'job': ['Programmer', 'Designer', 'Accountant']

})

#accessing the row using the iloc() and  loc() function
# When name is set as index
df = df.set_index('name')
df.loc['Alice'] #row Mike

# accessing (row) the data with the index, use iloc
df.iloc[1]

#accessing individual values
df.at['Alice', 'age']
df.iat[1, 0]

#changing individual values
df.at['Alice', 'age'] = 60

df.loc['Alice'] = [75, 'Clerk'] #Changing the age of Alice
df.loc['John'] = [90, 'Teacher'] # adding new row

# accessing the row from index 0:2
df.iloc[0:2]

# accessing row 0:3 and column 1
df.iloc[0:3, 1]

#--------------------------------------------------------------------------------------------------------------------

# Accessing a Column in a Dataframe
df

#Manipulating data (Applying Functions)
def myfunction(x):
    if x % 3 == 0:
        return x ** 2
    else:
        return x // 2

df.age.apply(myfunction)

def myfunction2(x):
    if x.endswith('r'):
        return 'without job'
    else:
        return x
    
df.job.apply(myfunction2)

df.at['Alice', 'age'] = float('nan') 


# # Iterate over dataframe
# # iterrows() is a method used to loop through a DataFrame row by row.
# for i, row in df.iterrows():
#     print(row['age'])

# for i, col in df.items():
#     print(col)




#--------------------------------------------------------------------------------------------------------------------
# Saving a file as pkl
#
import pickle

data = {
    "alpha" : [3, 5, 7],
    "beta" : [4, 5, 6]
}                           #Use curly braket {} for dictionary

with open('data.bin', 'wb') as f:   
    pickle.dump(data, f) # actual saving happens here, the dictionary 'data' is being saved in the file we just opened as f

# Note:
#1. instead of .bin, .pkl or .pickle can be used
#2. data.bin is the file being created
# 3. wb is the mode in which the data is being saved, that is in Write Binary Mode
# - w (Write): "Create a new file or empty the existing one."
# - b (Binary): "Don't treat this as text; handle it as raw data (0s and 1s)"

with open('data.bin', 'rb') as f:
    load_data = pickle.load(f)
    print(load_data) # return a dictionary

#-----------------------------------------------------------------------------
for i in range(1,11):
    print('PVLIT' + str(i))
