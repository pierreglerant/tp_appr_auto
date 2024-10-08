def dict_dict_first_key(dictionary, key):
    """
    TODO
    """
    return list(dictionary[key].keys())[0]
    

def lp_train_test_split(data, target_cols, ratio=0.3, seed=42):
    """
    TODO
    """
    filtered_data = data[data[target_cols].notna().all(axis=1)]
    test_data = filtered_data.sample(frac=ratio, random_state=seed)
    train_data = data.drop(test_data.index)

    X_train = train_data.drop(columns=target_cols)
    X_train = X_train.select_dtypes(include='Float64')
    Y_train = train_data[target_cols]

    X_test = test_data.drop(columns=target_cols)
    X_test = X_test.select_dtypes(include='Float64')
    Y_test = test_data[target_cols]

    return X_train, Y_train, X_test, Y_test
    