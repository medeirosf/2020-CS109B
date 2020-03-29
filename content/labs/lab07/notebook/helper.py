import numpy as np
import pandas as pd
import math

import random

def noisy_sin(steps_per_cycle = 50,
              number_of_cycles = 500,
              random_factor = 0.4):
    # Ref: 
    '''
    random_factor    : amount of noise in sign wave. 0 = no noise
    number_of_cycles : The number of steps required for one cycle
    
    Return : 
    pd.DataFrame() with column sin_t containing the generated sin wave 
    '''
    random.seed(0)
    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * random_factor))
    df["sin_t_clean"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)))
    return(df)


def load_data():
    steps_per_cycle = 10
    df = noisy_sin(steps_per_cycle=steps_per_cycle,
              random_factor = 0.02)
    
#     test_size = 0.25
#     ntr = int(len(df) * (1 - test_size))
#     df_train = df[["sin_t"]].iloc[:ntr] 
#     df_test  = df[["sin_t"]].iloc[ntr:] 
#     X_train = df_train.iloc[0:3749].values.reshape(-1,1,1)
#     y_train = df_train.iloc[1:3750].values.reshape(-1,1,1)

#     X_test = df_test.iloc[0:1250].values.reshape(-1,1,1)
#     y_test = df_test.iloc[1:1251].values.reshape(-1,1,1)

    def _load_data(data, n_prev = 100):  
        """
        data should be pd.DataFrame()
        """
        docX, docY = [], []
        for i in range(len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].values)
            docY.append(data.iloc[i+n_prev].values)
        alsX = np.array(docX)
        alsY = np.array(docY)

        return alsX, alsY

    length_of_sequences = 2
    test_size = 0.25
    ntr = int(len(df) * (1 - test_size))
    df_train = df[["sin_t"]].iloc[:ntr]
    df_test  = df[["sin_t"]].iloc[ntr:]
    (X_train, y_train) = _load_data(df_train, n_prev = length_of_sequences)
    (X_test, y_test)   = _load_data(df_test, n_prev = length_of_sequences)  

    return X_train, y_train, X_test, y_test
    

