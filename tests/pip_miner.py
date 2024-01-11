# THIS WOULD CONTAIN THE UTILITIES (FUNCTIONS, PARAMETERS) NECESSARY FOR THE RESEARCH
import matplotlib.pyplot as plt
import math # noqa
import mplfinance as mpf
import numpy as np
import pandas as pd
import xgboost as xgb

from hashlib import sha256 as shash
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder
from typing import Literal, List, Union

class PipMiner:

    def __init__(self, n_pivots:float, lookback : int, hold_period:int, n_clusters:int) -> None:
        self.n_pivots = n_pivots
        self.lookback = lookback
        self.hold_period = hold_period
        self.n_clusters = n_clusters
    
        # Training Data
        self._data = np.array([]) # Store the training data
        self._returns = np.array([]) # Store returns from each bar in training data
        
        self._unique_pivot_indices = [] # Store the indices where each patterns is found
        self._unique_pivot_patterns = [] # Store each unique patterns found
        self._unique_pivots = {} # Store pivots patterns and indices as key-value pairs

        self._cluster_model = None # Store the fitted clustering model used by the model
        self._clusters_pivot = [] # Store the pivot clusters
        self._clusters_indices = [] # Store the indices for the pivot clusters
        self._cluster_signals = []
        
        self._selected_long = [] # Cluster index for the best long cluster
        self._selected_short = [] # Cluster index for the best short cluster

        self._long_signal = None
        self._short_signal = None

        self._fit_martin = None
        self._perm_martins = []

        self._dist_measure = 3 # Select the distance measure for finding PIPs
        self._random_state = 14


    def train(self, data:np.ndarray, iterations:int=-1, cluster_method : Literal['kmeans', 'dbscan', 'spectral'] = 'spectral'):
        np.random.seed(self._random_state)

        self._data = data
        self._returns = pd.Series(data).diff().shift(-1) # Calculates the price change, and shifts it back to the original data (checking returns in future)

        self._find_unique_patterns() # Compute patterns from training

        # Cluster the patterns, and get their model instance
        if cluster_method == 'kmeans':
            self._cluster_patterns_kmeans(
            points=self._unique_pivot_patterns)
        elif cluster_method == 'dbscan':
            self._cluster_patterns_hdbscan(self._unique_pivot_patterns)
        else:
            # Default to 'spectral
            self._cluster_patterns_spectral(self._unique_pivot_patterns)

        self._assign_cluster_signals() # Assign signals (1s) for each clusters

        self._assess_clusters() # Assess clusters for longs and shorts

        self._fit_martin = self._compute_total_performance()

        if iterations < 1:
            return self._fit_martin

        # Start monte carlo permutation test
        data_copy = self._data.copy()
        
        for rep in range(1, iterations):
            x = np.diff(data_copy).copy() # Get the data changes 
            np.random.shuffle(x) # Shuffle the returns
            x = np.concatenate([np.array([data_copy[0]]), x])

            self._data = np.cumsum(x)
            self._returns = pd.Series(self._data).diff().shift(-1)

            print(f"Iteration {rep}") 

            # Same steps as above
            self._find_unique_patterns()
            self._cluster_patterns_kmeans(
                points=self._unique_pivot_patterns) # Cluster the patterns, and get their centers
            
            self._assign_cluster_signals() # Assign signals (1s) for each clusters
            self._assess_clusters() # Assess clusters for longs and shorts
            perm_martin = self._compute_total_performance()

            self._perm_martins.append(perm_martin) # Store permutation martin 


    def get_fit_martin(self):
        return self._fit_martin


    def get_permutation_martins(self):
        return self._perm_martins


    def plot_cluster_examples(self, dataframe: pd.DataFrame, cluster_i: int, grid_size: int = 5):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(grid_size, grid_size) # Create a grid of grid_size
        flat_axs = axs.flatten() # Flatten subplots (mak ethem accessible as 1D array instead of 2D)

        for i in range(len(flat_axs)): # Loop Through Each Subplot
            
            # If no available cluster example, break the loop
            if i >= len(self._clusters_indices[cluster_i]): 
                break
            
            # Get the indices for each pattern in the cluster
            pattern_index = self._clusters_indices[cluster_i][i]

            data_slice = dataframe.iloc[pattern_index - self.lookback + 1: pattern_index + 1]
            idx = data_slice.index
            
            # Get the Pivot Points
            pivot_indices, pivot_prices = self.find_pivots(data_slice['close'].to_numpy(), self.n_pivots, 3)
            pivot_lines = []
            colors = []

            for line_index  in range(self.n_pivots - 1):
                l0 = [(idx[pivot_indices[line_index]], pivot_prices[line_index]), (idx[pivot_indices[line_index + 1]], pivot_prices[line_index + 1])]
                pivot_lines.append(l0)
                colors.append('w')

            mpf.plot(data_slice, type='candle',alines=dict(alines=pivot_lines, colors=colors), ax=flat_axs[i], style='charles', update_width_config=dict(candle_linewidth=1.75) )
            flat_axs[i].set_yticklabels([])
            flat_axs[i].set_xticklabels([])
            flat_axs[i].set_xticks([])
            flat_axs[i].set_yticks([])
            flat_axs[i].set_ylabel("")

        fig.suptitle(f"Cluster {cluster_i}", fontsize=32)
        plt.show(), plt.clf()


    def plot_all_clusters(self):
        clusters = self._clusters_pivot
        print(f"Number of Clusters: {len(clusters)}")
        
        grid_size = 0

        # Get the optimal grid size
        for size in range(1, 10):
            grid_size = size
            if (size * size) >= len(clusters):
                break

        plt.style.use('dark_background')

        fig, axs = plt.subplots(grid_size, grid_size) # Create a grid of grid_size
        flat_axs = axs.flatten() # Flatten subplots (mak ethem accessible as 1D array instead of 2D)

        for cluster_index, cluster in enumerate(clusters):
            if cluster_index >= len(flat_axs):
                break
            
            flat_axs[cluster_index].set_title(f'Cluster {cluster_index+1} (members={len(cluster)})')

            # Plot each pattern in the cluster
            for pattern in cluster:
                flat_axs[cluster_index].plot(pattern)

            plt.grid(False)

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.suptitle('Line Plots of Clusters')
        
        plt.show(), plt.clf()


    def plot_cluster(self, index):
        clusters = self._clusters_pivot

        if len(clusters) <= index:
            raise ValueError('Cluster index out of bound.')
    
        plt.style.use('dark_background')

        # Plot each pattern in the cluster
        for pattern in clusters[index]:
            plt.plot(pattern)

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.suptitle('Line Plots of Clusters')
        
        plt.show(), plt.clf()


    # METHODS FOR TESTING THE MODEL
    def forward_test(self, data:np.ndarray, split_index=-1):
        _ret = pd.Series(data).diff().shift(-1) # Compute the returns of the data

        # Loop Through Data, and find the patterns
        last_pivot_indices = [0] * self.n_pivots # Create a list of zeros; this stores the last pivots patterns found.
        signals = [0] * len(data) # Initialize all signals as 0

        # Generate signals
        for index in range(self.lookback - 1, len(data)):
            # If signal at that index is not 0 (position is open), skip index
            if signals[index] != 0:
                continue

            # Get a window of data
            start_index = index - self.lookback + 1
            window = data[start_index:index + 1] # length would be self.lookback + 1
            
            pivot_indices, pivot_prices = self.find_pivots(window, self.n_pivots, dist_measure=self._dist_measure) # TODO : Implement other pivot algorithms
            pivot_indices = [pos + start_index for pos in pivot_indices]

            # Check internal pivots to see if it is the same as last (if they are on the same candles)
            same = pivot_indices[1: -1] == last_pivot_indices[1: -1]
            last_pivot_indices = pivot_indices 

            if same:
                # Only allow new unique patterns
                continue

            # Predict signal / generate signal 
            signal = self._predict_cluster(pivot_prices)

            if signal: 
                # If signal is 1 or -1, set the signal value for the next [holding_period] values
                for i in range(1, self.hold_period + 1):
                    if (index + i) >= len(signals):
                        break
                    signals[index + i] = signal

        # Calculate Returns trading the model
        returns = signals * _ret

        # Calculate the cumulative sum for the entire array
        log_return = np.cumsum(returns)

        # Plot the equity curve
        plt.plot(log_return, label='Log return curve')

        # Draw a vertical line at the split index
        plt.axvline(x=split_index, color='red', linestyle='--', label='Out-of-sample split')

        plt.xlabel('Trade Bars')
        plt.ylabel('Returns')
        plt.title('Log Returns')
        plt.legend()
        plt.show()


    def generate_signal(self, data:np.ndarray):
        '''
        Generates signal from a window of data
        '''
        # Generate Pivot Prices from
        _, pivot_prices = self.find_pivots(data, self.n_pivots, dist_measure=self._dist_measure) # TODO : Implement other pivot algorithms

        # Predict signal / generate signal 
        signal = self._predict_cluster(pivot_prices)

        return signal


    def _predict_cluster(self, points: Union[List[float], np.ndarray]):
        '''
        Categorizes a vertor into a cluster.

        Returns:
            signal (int) : Signal for the point vector
        '''
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if (not self._cluster_model):
            # Model has not been trained
            raise ValueError('Prediction was unsuccessfull. Class has not been trained.')

        points = self._normalize_points(points)

        
        cluster_index = self._cluster_model.predict(points.reshape(1, -1))

        if cluster_index in self._selected_long:
            return 1.0
        elif cluster_index in self._selected_short:
            return -1.0
        else:
            return 0.0


    # region - METHODS FOR PIVOTS FINDING
    # Find Perceptually Important Points in data
    def find_pivots(self, data: np.array, n_pivots: int, dist_measure: int):
        # dist_measure
        # 1 = Euclidean Distance
        # 2 = Perpindicular Distance
        # 3 = Vertical Distance

        pips_x = [0, len(data) - 1]  # Index
        pips_y = [data[0], data[-1]] # Price

        for curr_point in range(2, n_pivots):

            md = 0.0 # Max distance
            md_i = -1 # Max distance index
            insert_index = -1

            for k in range(0, curr_point - 1):

                # Left adjacent, right adjacent indices
                left_adj = k
                right_adj = k + 1

                time_diff = pips_x[right_adj] - pips_x[left_adj]
                price_diff = pips_y[right_adj] - pips_y[left_adj] + 1e-10
                slope = price_diff / time_diff
                
                intercept = pips_y[left_adj] - pips_x[left_adj] * slope

                for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                    
                    d = 0.0 # Distance
                    if dist_measure == 1: # Euclidean distance
                        d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                        d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                    elif dist_measure == 2: # Perpindicular distance
                        d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                    else: # Vertical distance    
                        d = abs( (slope * i + intercept) - data[i] )

                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            pips_x.insert(insert_index, md_i)
            pips_y.insert(insert_index, data[md_i])

        return pips_x, pips_y
    

    def _find_unique_patterns(self):
        self._unique_pivot_patterns.clear()
        self._unique_pivot_indices.clear()

        start_index = self.lookback - 1
        end_index = len(self._data)

        if not end_index:
            raise ValueError("Training data is not available.")

        # Loop Through Data, and find the patterns
        last_pivot_indices = [0] * self.n_pivots # Create a list of zeros; this stores the last pivots patterns found.

        for index in range(start_index, end_index):
            # Get a window of data
            start_index = index - self.lookback + 1
            window = self._data[start_index:
                                 index + 1] # length would be self.lookback + 1
            
            pivot_indices, pivot_prices = self.find_pivots(window, self.n_pivots, dist_measure=self._dist_measure) # TODO : Implement other pivot algorithms
            pivot_indices = [pos + start_index for pos in pivot_indices]

            # Check internal pivots to see if it is the same as last (if they are on the same candles)
            same = pivot_indices[1: -1] == last_pivot_indices[1: -1]
            
            if not same:
                # Z-Score normalize pattern
                pivot_prices = self._normalize_points(pivot_prices)
                self._unique_pivot_patterns.append(pivot_prices)
                self._unique_pivot_indices.append(index) # Store the bar index where pattern is found

                # Hash the pattern 
                key = self._hash_pattern(pivot_prices)
                self._unique_pivots[key] = index

            last_pivot_indices = pivot_indices        


    def _hash_pattern(self, point):
        if not isinstance(point, np.ndarray):
           point = np.array(point)

        sha = shash()
        sha.update(np.array(point).tobytes())

        return sha.hexdigest() 
        

    def _apply_constraints_to_patterns(self):
        i_pattern = 0
        while (i_pattern < len(self._unique_pivot_patterns)) and (len(self._unique_pivot_patterns) > 0):

            # Only allow patterns that have zigzag patterns
            pattern = self._unique_pivot_patterns[i_pattern]
            index = self._unique_pivot_indices[i_pattern]

            _dir = np.sign(pattern[1] - pattern[0])

            for i_pivot in range(2, self.n_pivots):
                _ = np.sign(pattern[i_pivot] - pattern[i_pivot -1])
                
                if _ == _dir:
                    # Same subsequent direction 
                    self._unique_pivot_patterns.remove(pattern)
                    self._unique_pivot_indices.remove(index)

                    i_pattern -= 1
                    break

                _dir = _

            i_pattern += 1


    def _normalize_points(self, points):
        points = (np.array(points) - np.mean(points)) / (np.std(points) + 1e-10)
        return np.array(points)
    # endregion


    # region - METHODS FOR CLUSTERING
    def _cluster_patterns_kmeans(self, points):
        self._clusters_pivot.clear()
        self._clusters_indices.clear()
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self._random_state, n_init='auto') # Initialize the KMeans model
        kmeans.fit(points) # Fit the model to your data

        # Extract clustering results: clusters and their centers
        self._clusters_pivot, self._clusters_indices = self._get_clusters(points, kmeans.labels_)
        
        # Train a XGBoost Classifier
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.n_clusters)
        model.fit(points, kmeans.labels_)
        
        # Initialize the clustering model
        self._cluster_model = model


    def _cluster_patterns_spectral(self, points):
        self._clusters_pivot.clear()
        self._clusters_indices.clear()
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Ensure n_neighbors is not greater than n_samples
        n_neighbors = min(self.n_clusters, points.shape[0] - 1)

        spectral = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=self._random_state, n_jobs=-1)
        spectral.fit(points) # Fit the model to your data

        # Extract clustering results: clusters and their centers
        self._clusters_pivot, self._clusters_indices = self._get_clusters(points, spectral.labels_)

        # Train a XGBoost Classifier
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.n_clusters)
        model.fit(points, spectral.labels_)

        # Initialize the clustering model
        self._cluster_model = model
    

    def _cluster_patterns_hdbscan(self, points):
        self._clusters_pivot.clear()
        self._clusters_indices.clear()
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Initialize the HDBSCAN model
        hdbscan = HDBSCAN(n_jobs=-1)
        hdbscan.fit(points)  # Fit the model to your data

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(hdbscan.labels_)

        # Extract clustering results: clusters and their centers
        self._clusters_pivot, self._clusters_indices = self._get_clusters(points, encoded_labels)

        # Train a XGBoost Classifier
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(encoded_labels)))
        model.fit(points, encoded_labels)

        # Initialize the clustering model
        self._cluster_model = model


    def _get_clusters(self, points, cluster_labels:list[int]):
        n_clusters = cluster_labels.max() + 1
        clusters = {key : [] for key in range(n_clusters)}
        indices = [] # Store the indices of each pattern in a cluster

        for i in range(len(points)):
            label = cluster_labels[i]
            point = points[i]
            
            clusters[label].append(point) # Add the point at its corresponding cluster label

        for cluster in clusters.values():
            # Get the index for each pattern in every clusters 
            _indices = [self._unique_pivots[self._hash_pattern(pattern)] for pattern in cluster]
            indices.append(_indices)

        return list(clusters.values()), indices

    #endregion


    # region - METHODS FOR SIGNALS
    def _assign_cluster_signals(self):
        self._cluster_signals.clear()

        for cluster in self._clusters_pivot: # Loop through each cluster
            signal = np.zeros(len(self._data)) # Initialise the signal array; value is 1 for the self.hold_period after the signal
            for pattern in cluster: # Loop through each pattern in cluster
                key = self._hash_pattern(pattern)
                bar_index = self._unique_pivots.get(key)
                
                # Fill signal with 1s following pattern identification
                # for hold period specified
                signal[bar_index: bar_index + self.hold_period] = 1. 
            
            self._cluster_signals.append(signal)

    
    def _assess_clusters(self):
        self._selected_long.clear()
        self._selected_short.clear()
        
        # Assign clusters to long/short/neutral
        cluster_martins = []
        for clust_i in range(len(self._clusters_pivot)): # Loop through each cluster
            sig = self._cluster_signals[clust_i] # Get signal array for clusters 
            sig_ret = self._returns * sig # Calculate the returns for each clusters
            martin = self._compute_martin(sig_ret) # Calculate the martin ratio for each cluster. TODO : Apply other metrics
            cluster_martins.append(martin) # Store the martin ratio

        best_long = np.argmax(cluster_martins) # Get index of cluster with best long results
        best_short = np.argmin(cluster_martins) # Get index of cluster with best short results
  
        self._selected_long.append(best_long) # Cluster index for the best long cluster
        self._selected_short.append(best_short) # Cluster index for the best short cluster


    def _compute_total_performance(self):
        long_signal = np.zeros(len(self._data))
        short_signal = np.zeros(len(self._data))

        for clust_i in range(len(self._clusters_pivot)):
            if clust_i in self._selected_long:
                long_signal += self._cluster_signals[clust_i]
            elif clust_i in self._selected_short:
                short_signal += self._cluster_signals[clust_i]
        
        long_signal /= len(self._selected_long)
        short_signal /= len(self._selected_short)
        short_signal *= -1

        self._long_signal = long_signal
        self._short_signal = short_signal
        rets = (long_signal + short_signal) * self._returns

        martin = self._compute_martin(rets)
        return martin


    def _compute_martin(self, rets: np.array):
        rsum = np.sum(rets)
        short = False
        if rsum < 0.0:
            rets *= -1
            rsum *= -1
            short = True

        csum = np.cumsum(rets)
        eq = pd.Series(np.exp(csum))
        sumsq = np.sum( ((eq / eq.cummax()) - 1) ** 2.0 )
        ulcer_index = (sumsq / len(rets)) ** 0.5
        martin = rsum / (ulcer_index + 1.e-10)
        if short:
            martin = -martin

        return martin

    # endregion


if __name__ == '__main__':

    # Read In Full Data
    # data = pd.read_parquet('/Users/jerryinyang/Code/quantbt/data/prices/BTCUSDT.parquet')
    data = pd.read_csv('/Users/jerryinyang/Code/quantbt/data/prices/AAPL.US_H1.csv')
    data.set_index(data['datetime'], inplace=True)
    data.index = pd.to_datetime(data.index)

    data.columns = data.columns.str.lower()

    # Define your date range
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    # Filter the DataFrame
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    x = np.log(data['close'].to_numpy())
    
    split_index = int(round(0.6 * len(x)))
    x_train = x[:split_index]

    n_pivots = 5
    lookback = 20
    hold_period = 6
    n_clusters = 85 # math.factorial(n_pivots)

    miner = PipMiner(n_pivots=n_pivots, lookback=lookback, hold_period=hold_period, n_clusters=n_clusters)
    miner.train(x_train)
    # miner.plot_all_clusters()

    # Forward Test
    miner.forward_test(x, split_index)

    
    # Monte Carlo test, takes about an hour..
    # miner.train(x, iterations=100)
    
    # plt.style.use('dark_background')
    # actual_martin = miner.get_fit_martin()
    # perm_martins = miner.get_permutation_martins()
    # ax = pd.Series(perm_martins).hist()
    # ax.set_ylabel("# Of Permutations")
    # ax.set_xlabel("Martin Ratio")
    # ax.set_title("Permutation's Martin Ratio BTC-USDT 1H 2018-2020")
    # ax.axvline(actual_martin, color='red')
    # plt.show(), plt.clf()