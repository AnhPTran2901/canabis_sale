from preprocessing import features_to_scale, features, target, create_feature, prep_data
import pandas as pd
import pickle

def read_data(file_path):
    df = pd.read_csv(file_path)
    df['Week Ending'] = pd.to_datetime(df['Week Ending'])
    return df

def batch_sale_prediction(file_path, model_path):
    df = read_data(file_path)

    new_data = create_feature(df)
    X_new, _ = prep_data(new_data, features, features_to_scale, target)

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X_new)

    return predictions