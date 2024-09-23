# #read pickle file
# import pickle
# import numpy as np
# path = "/home/vuonghn/research/code/Agriculture/DeepSatModels/datasets/AR24/pickle24x24"
# import os
# list_file = os.listdir(path)
# for file in list_file:
#     file_path = os.path.join(path, file)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     print("label shape ", data['labels'].shape, np.unique(data['labels']))

import pandas as pd

# Sample requirements
sample_requirements = {
    1: 1,  # January
    2: 1,  # February
    3: 1,  # March
    4: 2,  # April
    5: 2,  # May
    6: 2,  # June
    7: 2,  # July
    8: 2,  # August
    9: 2,  # September
    10: 1, # October
    11: 1, # November
    12: 1  # December
}

def resample_dates(dates):
    # Convert dates to DataFrame
    df = pd.DataFrame(dates, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    resampled_dates = []
    
    # Group by month and sample
    for month, group in df.groupby('month'):
        sample_size = sample_requirements.get(month, 0)
        if sample_size > 0:
            group = group.sort_values(by='date')
            if sample_size == 1:
                # Prefer the date in the middle of the month
                middle_date = group.iloc[(group['day'] - 15).abs().argsort()[:1]]
                resampled_dates.extend(middle_date['date'].dt.strftime('%Y-%m-%d').tolist())
            elif sample_size == 2:
                # Prefer the beginning and ending of the month
                beginning_date = group.iloc[:1]
                ending_date = group.iloc[-1:]
                resampled_dates.extend(beginning_date['date'].dt.strftime('%Y-%m-%d').tolist())
                resampled_dates.extend(ending_date['date'].dt.strftime('%Y-%m-%d').tolist())
    
    return resampled_dates

# Example usage
dates = [
    '2023-02-12', '2023-11-04', '2023-05-03', '2023-10-30', '2023-11-01', 
    '2023-06-27', '2023-09-25', '2023-07-27', '2023-09-27', '2023-03-04', 
    '2023-10-07', '2023-09-10', '2023-06-17', '2023-04-08', '2023-09-30', 
    '2023-10-25', '2023-09-15', '2023-08-18', '2023-10-17', '2023-01-08', 
    '2023-01-23', '2023-04-18', '2023-04-10', '2023-03-26', '2023-06-04', 
    '2023-03-19', '2023-03-14', '2023-04-23', '2023-06-09', '2023-05-20', 
    '2023-05-28', '2023-08-31', '2023-06-29', '2023-08-21', '2023-05-25', 
    '2023-10-12', '2023-03-29', '2023-08-28', '2023-01-05', '2023-12-11', 
    '2023-12-19', '2023-07-17', '2023-11-11', '2023-06-24', '2023-01-15', 
    '2023-01-10', '2023-05-18', '2023-02-27', '2023-12-26', '2023-12-06', 
    '2023-08-11', '2023-06-07', '2023-05-30', '2023-09-17', '2023-10-22', 
    '2023-10-02', '2023-10-20', '2023-02-17', '2023-07-02', '2023-12-14', 
    '2023-10-10', '2023-09-07', '2023-01-20'
]

# Sort dates
# dates.sort()
print(dates)
resampled = resample_dates(dates)
resampled.sort()
print(resampled)