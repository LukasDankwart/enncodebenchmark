import joblib
import numpy as np

if __name__ == '__main__':
    my_scaler = joblib.load('min-max-sclaler.gz')

    ctx = [[0.39827148, 0.57651246, 0.62187576, 0.5454163 ]] # [[0.48124388, 0.3455516,  0.3959416,  0.1397533 ]]
    original = [[0.5967407,  0.57651246, 0.62187576, 0.5454163 ]] # [[0.5558519, 0.3455516, 0.3959416, 0.1397533]]


    inv_ctx = my_scaler.inverse_transform(ctx)
    print(inv_ctx)
    inv_original = my_scaler.inverse_transform(original)
    print(inv_original)

    print(my_scaler.n_features_in_)  # Works if my_scaler is a fitted sklearn scaler