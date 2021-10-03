
def select_features(dataset, target_var, num_features=10):

    features = list(dataset.corr().abs().nlargest(num_features, target_var).index)

    # remove duplicates
    features = list(set(features))

    return features
    