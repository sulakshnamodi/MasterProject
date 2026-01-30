import os, sys
import pandas as pd
import numpy as np
import pickle
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sn


# setting of the subdataset
subdataset_root = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\data\preprocessed\subdataset1'
subdataset_filename = 'piaac_norway_subdataset1.pkl'
subdataset_filepath = os.path.join(subdataset_root, subdataset_filename)
outputfolder = r'G:\My Drive\Sulakshna\Sulakshna Drive\Codes\MasterProject\results\regression'

# load the subdataset (pkl file)
with open(subdataset_filepath, 'rb') as f:
    loaded_data = pickle.load(f)

subdataset_df = loaded_data['dataframe']

# Create the strata before entering the loops
# GENDER_R (1,2), ED_GROUP (0,1), PAREDC2 (1,2,3)
maihda_df = subdataset_df.dropna(subset=['GENDER_R', 'ED_GROUP', 'PAREDC2']).copy()

# Create a combined ID for the 12 possible intersections
maihda_df['strata'] = maihda_df.groupby(['GENDER_R', 'ED_GROUP', 'PAREDC2']).ngroup()

vpc_list = []
strata_var_list = []

for i in range(1, 11):
    current_score = f'PVLIT{i}'
    
    # Fit the Null Multilevel Model
    # Outcome ~ 1 (Intercept), grouped by 'strata'
    # we use SPFWT0 as the weight
    md = smf.mixedlm(f"{current_score} ~ 1", maihda_df, 
                     groups=maihda_df["strata"])
    mdf = md.fit()

    # Extract Variances
    # sigma2_g = between-strata variance
    # sigma2_e = within-strata (residual) variance
    sigma2_g = mdf.cov_re.iloc[0, 0]
    sigma2_e = mdf.scale
    
    vpc = sigma2_g / (sigma2_g + sigma2_e)
    
    vpc_list.append(vpc)
    strata_var_list.append(sigma2_g)

# Final MAIHDA Summary
avg_vpc = np.mean(vpc_list)
print(f"Average Intersectional VPC across 10 PVs: {avg_vpc:.2%}")

# Get the Best Linear Unbiased Predictors (BLUPs) from the last model
random_effects = mdf.random_effects

# Convert to a DataFrame for plotting
re_df = pd.DataFrame.from_dict(random_effects, orient='index').reset_index()
re_df.columns = ['strata', 'Effect']

# Map the strata IDs back to their names for the plot labels
names = maihda_df.groupby('strata')[['GENDER_R', 'ED_GROUP', 'PAREDC2']].first()
re_df = re_df.merge(names, on='strata')

# Plot the 'Intersectional Effects'
plt.figure(figsize=(12, 6))
sn.barplot(data=re_df.sort_values('Effect'), x='Effect', y='strata', orient='h', palette='vlag')
plt.title("Intersectional Effects: Deviation from Norway Average Literacy")
plt.xlabel("Points above/below average")
plt.show()