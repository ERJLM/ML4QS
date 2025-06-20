import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import butter, lfilter, filtfilt
from sklearn.decomposition import PCA



def remove_outliers(df):
    """
    Removes the outliers using Local Outlier Factor.

    Args:
        df (pd.Dataframe): Dataframe with outliers.
    """
    
    # Create new auxiliar columns
    df['outlier'] = 0
    df['lof_score'] = 0
    
    # Apply LOF to detect the outliers
    for user in df.user.unique():
        X = df[df.user == user].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        clf = LocalOutlierFactor(n_neighbors=20)
        clf.fit_predict(X_scaled)
        lof_scores = -clf.negative_outlier_factor_
        df.loc[X.index, 'lof_score'] = lof_scores
    
    # Mark the outliers
    df.loc[df.lof_score > 1.1, 'outlier'] = 1
    df = df.drop(columns=['lof_score'])
    
    # Remove all the outliers so we can impute later
    df.loc[df.outlier == 1, df.columns.difference(['seconds_elapsed', 'user'])] = np.nan
    df = df.drop(columns="outlier")
    
    return df
    
def impute(df):
    """
    Impute the missing values using interpolation.

    Args:
        df (pd.Dataframe): Dataframe with missing values.
    """
    for user in df.user.unique():
        # Fill the missing values using interpolation
        df[df.user == user] = df[df.user == user].interpolate()
        # Fill the initial data points
        df[df.user == user] = df[df.user == user].fillna(method='bfill')
    
    return df


def low_pass_filter_aux(data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
        # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype='low', output='ba', analog=False)
        if phase_shift:
            data_table[col + '_lowpass'] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + '_lowpass'] = lfilter(b, a, data_table[col])
        return data_table
    
def low_pass_filter(df):
    
    for col in df.columns.to_list():
        if col in ["user", "seconds_elapsed"]:
            continue
        
        for user in df.user.unique():
            user_mask = df.user == user
            user_df = df[user_mask]
            user_df = low_pass_filter_aux(df, col, sampling_frequency=100, cutoff_frequency=0.5, order=3)
            
    return df

def pca(df):
    """
    Apply PCA to the dataframe.

    Args:
        df (pd.Dataframe): Dataframe.
    """
    X = df.drop(columns=['user'])
    
    # First we normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use PCA with the same number of components as the number of columns (excluding the target).
    pca = PCA(n_components= len(X.columns))
    pca.fit(X_scaled)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a column for each component
    pca_columns = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)

    # Concatenate with original dataframe
    df = pd.concat([df, df_pca], axis=1)
    
    return df
        
            

