
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import pickle
from preprocessing import features, features_to_scale, target, create_feature, prep_data


class MLTechniques:
    """Machine Learning techniques for regression (XGBoost)"""

    def optimize_xgb(X_train, y_train, X_test, y_test):
        space = {
            'max_depth': hp.choice('max_depth', range(1, 11)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'gamma': hp.uniform('gamma', 0, 1),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        }

        def objective(params):
            model = xgb.XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                gamma=params['gamma'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                
            )
     
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            return {'loss': mse, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)

        return best

    def xgb_teq(X_train, y_train, X_test, y_test, n_splits=5):
        """
        Performs K-fold CV over a given training set and test set for XGBoost
        and outputs rmse metrics and prediction/true label outputs.
        Args:
            X_train (np.array): Full training set.
            y_train (np.array): Full training set labels.
            X_test (np.array): Full test set.
            y_test (np.array): Full test set labels.
            n_splits (int): Number of K-fold splits for K-fold CV.

        Returns:
            ts_xgb (list): List of mean square errors from each training set k fold split.
            cvs_xgb (list): List of mean square errors from each validation set k fold split.
            mse_xgboost (float): Entire training set List of mean square errors score.
            predictions_test_set_xgb (pd.DataFrame): Single-column dataframe of test set predictions.
            preds (pd.DataFrame): Two-column dataframe of test set true labels and predictions.
            final_model (object): XGBoost train object.
        """
        # Split the data into training and test sets

        best_params = MLTechniques.optimize_xgb(X_train, y_train, X_test, y_test)
        param = {
            'max_depth': best_params['max_depth'],
            'learning_rate': best_params['learning_rate'],
            'n_estimators': best_params['n_estimators'],
            'gamma': best_params['gamma'],
            'min_child_weight': best_params['min_child_weight'],
            'subsample':best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'objective':'reg:squarederror'
            
        }

        # Make a data frame of the size of y_train with one column - 'prediction'. Then make a Stratified
        # K fold object with n_splits
        ts_xgb = []
        cvs_xgb = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_index, cv_index in k_fold.split(
           X_train
        ):

            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                X_train.iloc[train_index, :],
                X_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]


            dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dcv_fold = xgb.DMatrix(X_cv_fold)

           
            # Use best params and the dtrain dataset to perform K fold cross validation on the (already k-folded) dataset.
            cv_results = xgb.cv(
                param,
                dtrain_fold,
                num_boost_round=50,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=5,
                metrics='rmse',
               
            )
            best_nrounds = cv_results.shape[0]
            print(
                np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)
            )
            # Train on dtrain
            model = xgb.train(
                param,
                dtrain_fold,
                num_boost_round=best_nrounds,
            )


            # Calculate training msqe between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            mse_training = mean_squared_error(
                y_train_fold, model.predict(dtrain_fold)
            )
            ts_xgb.append(mse_training)

            # Predict on the val set and insert the predictions from this fold into the dataframe created outside the loop
            predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"] = model.predict(dcv_fold)

            # Calculate val set log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold msqe values
            mse_cv = mean_squared_error(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_xgb.append(mse_cv)

        # Now that we have looped through all folds, calculate the total log loss across all data
        mse_xgboost = mean_squared_error(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        ## Train final model on entire training set

        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_results = xgb.cv(
            param,
            dtrain,
            num_boost_round=50,
            nfold=5,
            early_stopping_rounds=10,
            verbose_eval=5,
            metrics='rmse',
            as_pandas=True
        )
        best_nrounds_final = cv_results.shape[0]
        final_model = xgb.train(
            param,
            dtrain,
            num_boost_round=best_nrounds_final,
        )

        # Make a dmatrix out of the test set, and predict on it, then store it in a dataframe.
        dtr = xgb.DMatrix(data=X_test, label=y_test)
        predictions_test_set_xgb = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_xgb.loc[:, "prediction"] = final_model.predict(dtr)
        mse_xgboost_test = mean_squared_error(
            y_test, predictions_test_set_xgb.loc[:, "prediction"]
        )

        return (
            ts_xgb,
            cvs_xgb,
            mse_xgboost,
            predictions_test_set_xgb,
            preds,
            mse_xgboost_test,
            final_model,
        )
    
    def log_xgb_model(x_train, y_train, x_test, y_test):
        mlflow.set_experiment("XGBoost Regression")
        
        # We can assign run_id to keep track of different runs
        with mlflow.start_run():
            ts_xgb, cvs_xgb, mse_xgboost, predictions_test_set_xgb, preds, mse_xgboost_test, final_model = MLTechniques.xgb_teq(
                x_train, 
                y_train, 
                x_test, 
                y_test)
        
            
            # Log metrics
            mlflow.log_metric("training mse", mse_xgboost)
            mlflow.log_metric("rmse", np.sqrt(mse_xgboost))
            mlflow.log_metric("mean_cv_score", np.mean(cvs_xgb))
            mlflow.log_metric("test mse", mse_xgboost_test)
            
            # Log the model
            mlflow.xgboost.log_model(final_model, "xgboost_model")
            
            return ts_xgb, cvs_xgb, mse_xgboost, predictions_test_set_xgb, preds, mse_xgboost_test, final_model
    

        
if __name__ == "__main__":
    # Usage
    print("read the data")
    df = pd.read_csv("./data/Cannabis_Retail_Sales_by_Week_Ending.csv")
    df['Week Ending'] = pd.to_datetime(df['Week Ending'])
    
    X, y = prep_data(df, features, features_to_scale, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    ts_xgb, cvs_xgb, mse_xgboost, predictions_test_set_xgb, preds, mse_xgboost_test, final_model = MLTechniques.log_xgb_model(
        X_train, 
        y_train, 
        X_test, 
        y_test)

    # Save the best model. 
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    # Ater running, the scrip, you can start mlflow UI to view the result:
    # mlflow ui
    


