##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import scipy
import math
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from scipy.spatial import distance as ssd

# Helper functions to replace util.distance and util.normalize_dataset

def compute_pairwise_distances(data, metric):
    """
    Compute pairwise distances for the given data array using scipy
    """
    # data: DataFrame or 2D array
    arr = np.asarray(data)
    return ssd.squareform(ssd.pdist(arr, metric=metric))


def normalize_dataset(df, cols):
    """
    Min-max normalize specified columns in a DataFrame
    """
    norm_df = df.copy()
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            norm_df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            norm_df[col] = 0.0
    return norm_df


# Class for outlier detection algorithms based on some distribution of the data.
class DistributionBasedOutlierDetection:

    def chauvenet(self, data_table, col, C):
        # Compute the mean and standard deviation.
        mean = data_table[col].mean()
        std = data_table[col].std()
        N = len(data_table)
        criterion = 1.0 / (C * N)

        deviation = abs(data_table[col] - mean) / std
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)

        mask = []
        for i in range(N):
            prob = 1.0 - 0.5 * (scipy.special.erf(high.iloc[i]) - scipy.special.erf(low.iloc[i]))
            mask.append(prob < criterion)

        data_table[col + '_outlier'] = mask
        return data_table

    def mixture_model(self, data_table, col):
        data = data_table[data_table[col].notnull()][col]
        g = GaussianMixture(n_components=3, max_iter=100, n_init=1)
        reshaped = data.values.reshape(-1, 1)
        g.fit(reshaped)
        probs = g.score_samples(reshaped)

        df_probs = pd.DataFrame(
            np.power(10, probs),
            index=data.index,
            columns=[col + '_mixture']
        )
        data_table = pd.concat([data_table, df_probs], axis=1)
        return data_table


# Class for distance based outlier detection.
class DistanceBasedOutlierDetection:

    def distance_table(self, data_table, cols, metric):
        subset = data_table[cols].astype('float32')
        dist_matrix = compute_pairwise_distances(subset, metric)
        return pd.DataFrame(dist_matrix, index=data_table.index, columns=data_table.index)

    def simple_distance_based(self, data_table, cols, d_function, dmin, fmin):
        new_data_table = normalize_dataset(data_table.dropna(subset=cols), cols)

        # Create the distance table first between all instances:
        self.distances = self.distance_table(new_data_table, cols, d_function)

        mask = []
        # Pass the rows in our table.
        for i in range(0, len(new_data_table.index)):
            # Check what faction of neighbors are beyond dmin.
            frac = (float(sum([1 for col_val in self.distances.iloc[i, :].tolist(
            ) if col_val > dmin]))/len(new_data_table.index))
            # Mark as an outlier if beyond the minimum frequency.
            mask.append(frac > fmin)
        if data_table.get('simple_dist_outlier') is None:
            data_mask = pd.DataFrame(mask, index=new_data_table.index, columns=[
                                     'simple_dist_outlier'])
            data_table = pd.concat([data_table, data_mask], axis=1)
        else:
            data_table['simple_dist_outlier'] = pd.Series(mask, index=new_data_table.index)
        del self.distances
        return data_table

    def local_outlier_factor(self, data_table, cols, metric, k):
        new_dt = normalize_dataset(data_table.dropna(subset=cols), cols)
        self.distances = self.distance_table(new_dt, cols, metric)

        lof_scores = []
        for i in range(len(new_dt)):
            lof_scores.append(self.local_outlier_factor_instance(i, k))

        lof_col = 'lof'
        lof_df = pd.DataFrame(lof_scores, index=new_dt.index, columns=[lof_col])
        data_table = pd.concat([data_table, lof_df], axis=1)
        del self.distances
        return data_table

    def reachability_distance(self, k, i1, i2):
        k_dist, _ = self.k_distance(i2, k)
        return max(k_dist, self.distances[i1, i2])

    HIGH_VALUE = 1e10

    def local_reachability_density(self, i, k, k_dist_i, neighbors):
        reach_dists = [self.reachability_distance(k, i, nbr) for nbr in neighbors]
        if sum(reach_dists) == 0:
            return float(self.HIGH_VALUE)
        return len(neighbors) / sum(reach_dists)

    def k_distance(self, i, k):
        dists = np.array(self.distances[i])
        idx = np.argpartition(dists, k+1)[:k+1]
        neighbors = [j for j in idx if j != i]
        return dists[neighbors].max(), neighbors

    def local_outlier_factor_instance(self, i, k):
        k_dist_i, neighbors = self.k_distance(i, k)
        lrd_i = self.local_reachability_density(i, k, k_dist_i, neighbors)

        ratios = []
        for nbr in neighbors:
            k_dist_n, nbr_neighbors = self.k_distance(nbr, k)
            lrd_n = self.local_reachability_density(nbr, k, k_dist_n, nbr_neighbors)
            ratios.append(lrd_n / lrd_i)

        return np.mean(ratios)
