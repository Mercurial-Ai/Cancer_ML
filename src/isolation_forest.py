from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

def isolation_forest(features, target):

    isolated_forest=RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=6) 

    isolated_forest.fit(features, target)
    predicted = isolated_forest.predict(features)

    predicted_df = pd.DataFrame(predicted)
    predicted_df.to_csv('data_anomaly.csv')

    return predicted
