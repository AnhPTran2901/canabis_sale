import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# Usage:
features_to_scale = ['Year',"Dayofmonth", "Dayofyear", 'Adult-Use Average Product Price', 'Medical Average Product Price']

features = ['Week', "Dayofweek", "Dayofmonth", "Dayofyear",
            'Year', 'Quarter', 'Adult-Use Average Product Price', 'Medical Average Product Price']

target = 'Medical Marijuana Retail Sales'

def create_feature(df):
    df['Week'] = df['Week Ending'].dt.isocalendar().week
    df['Year'] = df['Week Ending'].dt.year
    df['Quarter'] = df['Week Ending'].dt.quarter
    df['Month'] = df['Week Ending'].dt.month
    df["Dayofweek"] = df['Week Ending'].dt.dayofweek
    df["Dayofmonth"] = df['Week Ending'].dt.day
    df["Dayofyear"] = df['Week Ending'].dt.dayofyear

    X = df[features]
    y = df[target]
    return df


def prep_data(df, features, features_to_scale, target):
    ## Feature Scaling 
    df = create_feature(df)

    if len(features_to_scale) > 0:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    else:
        df_scaled = df.copy()
        scaler = 'No scaler needed'
    
    X = df_scaled[features]
    y = df_scaled[target]
    
    return X,y