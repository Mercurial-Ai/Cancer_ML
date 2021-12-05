from sklearn.ensemble import IsolationForest
import pandas as pd 

def isolation_forest(features, target):

    isolated_forest=IsolationForest(n_estimators=100, n_jobs=-1,random_state=42) 

    predicted = isolated_forest.fit_predict(features, target)

    predicted_df = pd.DataFrame(predicted)
    predicted_df.to_csv('data_anomaly.csv')

    return predicted
