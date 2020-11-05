import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score


def load_data(output_path = '../Data/nao_index_train.npy',
              x1_path = '../Data/tas_train.npy',
              x2_path = '../Data/psl_train.npy'):
    '''
    Parameters
    ----------
    output_path : TYPE, optional
        DESCRIPTION. The default is './Data/nao_index_train.npy'.
    x1_path : TYPE, optional
        DESCRIPTION. The default is './Data/tas_train.npy'.
    x2_path : TYPE, optional
        DESCRIPTION. The default is './Data/psl_train.npy'.

    Returns
    -------

    x1 : North Atlantic and Tropical Atlantic near surface air temperature, 
        October-November average (one vector of dimension M1 per year= instance)
    x2 : North Atlantic sea-level-pressure, October-November average 
        (one vector of dimension M2 per instance), Shape(N,M2)
    y : North Atlantic Oscillation Index

    '''
    
    y = np.load(output_path)
    x1 = np.load(x1_path)
    x2 = np.load(x2_path)
    return x1, x2, y




x1, x2, y = load_data()
pca = PCA(n_components=50)

pca.fit(x1)
x1_reduced = pca.transform(x1)

pca.fit(x2)
x2_reduced = pca.transform(x2)

x_combined = np.concatenate((x1_reduced,x2_reduced), axis=1)


X_train, X_test, y_train, y_test = train_test_split(x_combined, y, test_size=0.12, random_state=123)

reg = LinearRegression().fit(X_train, y_train)

train_pred = reg.predict(X_train)
test_pred = reg.predict(X_test)

print('MSE train: %.3f'%np.mean(np.square(train_pred-y_train)))
print('MSE test: %.3f'%np.mean(np.square(test_pred-y_test)))

def to_binary(y_in):
    y_out = y_in>0
    y_out = y_out.astype(float)
    return y_out


train_pred_bin = to_binary(train_pred)
test_pred_bin = to_binary(test_pred)
train_y_bin = to_binary(y_train)
test_y_bin = to_binary(y_test)

print('The train accuracy: %.3f'%accuracy_score(train_pred_bin, train_y_bin))
print('The test accuracy: %.3f'%accuracy_score(test_pred_bin, test_y_bin))

# =============================================================================
# MSE train: 0.869
# MSE test: 1.055
# The train accuracy: 0.628
# The test accuracy: 0.481
# =============================================================================

