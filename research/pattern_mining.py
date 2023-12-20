# THIS WOULD CONTAIN THE UTILITIES (FUNCTIONS, PARAMETERS) NECESSARY FOR THE RESEARCH
from hashlib import sha256 as shash # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # noqa
from utils import debug, clear_terminal # noqa

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find Perceptually Important Points in data
def find_pips(data: np.array, n_pips: int, dist_measure: int):
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance

    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
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


class PatternMiner:

    def __init__(self, n_pivots:float, lookback : int, hold_period:int, random_state=14) -> None:
        self.n_pivots = n_pivots
        self.lookback = lookback
        self.hold_period = hold_period
        self.random_state = random_state

        # Training Data
        self._data = np.array([]) # Store the training data
        self._returns = np.array([]) # Store returns from each bar in training data
        
        self._unique_pivot_indices = [] # Store the indices where each patterns is found
        self._unique_pivot_patterns = [] # Store each unique patterns found
        self._unique_pivots = {} # Store pivots patterns and indices as key-value pairs

        self._cluster_centers = [] # Store the cluster centers
        self._clusters_pivot = [] # Store the pivot clusters
        self._clusters_indices = [] # Store the indices for the pivot clusters
        
        self._cluster_signals = []

        self._dist_measure = 3 # Select the distance measure for finding PIPs


    def train(self, data:np.ndarray):
        self._data = data
        self._returns = pd.Series(data).diff().shift(-1) # Calculates the price change, and shifts it back to the original data (checking returns in future)

        self._find_unique_patterns() # Compute patterns from training

        # Find Optimal K value for KMeans clustering
        # Using the silhouette method, we get the index of the k value with the highest silhouette score
        # Then, we add 1 to get the k value 
        n_clusters = np.argmax(self._ksearch_silhouette(self._unique_pivot_patterns, kmax=30)) + 1
        self._cluster_patterns_kmeans(
            points=self._unique_pivot_patterns, 
            n_clusters=n_clusters) # Cluster the patterns, and get their centers
        
        self._assign_cluster_signals() # Assign signals (1s) for each clusters

        return (len(self._unique_pivot_patterns))


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
            
            pivot_indices, pivot_prices = find_pips(window, self.n_pivots, dist_measure=self._dist_measure) # TODO : Implement other pivot algorithms
            pivot_indices = [pos + start_index for pos in pivot_indices]

            # Check internal pivots to see if it is the same as last (if they are on the same candles)
            same = pivot_indices[1: -1] == last_pivot_indices[1: -1]
            
            if not same:
                # Z-Score normalize pattern
                pivot_prices = list((np.array(pivot_prices) - np.mean(pivot_prices)) / np.std(pivot_prices))
                self._unique_pivot_patterns.append(pivot_prices)
                self._unique_pivot_indices.append(index) # Store the bar index where pattern is found

                # Hash the pattern 
                key = self._hash_pattern(pivot_prices)
                self._unique_pivots[key] = index

            last_pivot_indices = pivot_indices          
        
    
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


    def _hash_pattern(self, point):
        if not isinstance(point, np.ndarray):
           point = np.array(point)

        return shash(np.array(point).tobytes()).digest()


    # METHODS FOR CLUSTERING
    def _get_clusters(self, points, n_clusters, cluster_labels:list[int]):
        clusters = [[]] * n_clusters

        for i in range(len(points)):
            label = cluster_labels[i]
            point = points[i]
            
            clusters[label].append(point) # Add the point at its corresponding cluster label

        return clusters
        

    def _cluster_patterns_kmeans(self, points, n_clusters:int):
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto') # Initialize the KMeans model
        kmeans.fit(points) # Fit the model to your data

        # Extract clustering results: clusters and their centers
        self._clusters_pivot = self._get_clusters(points, n_clusters, kmeans.labels_)
        self._cluster_centers = kmeans.cluster_centers_


    def _ksearch_elbow(self, points, kmax):
        sse = []
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters = k, n_init='auto').fit(points)
            
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0
            
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
                
            sse.append(curr_sse)
        return sse


    def _ksearch_silhouette(self, points, kmax):
        sil = []
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters = k, n_init='auto').fit(points)
            labels = kmeans.labels_
            sil.append(silhouette_score(points, labels, metric = 'euclidean'))
            
        return sil


    # METHODS FOR SIGNALS
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



if __name__ == '__main__':
    clear_terminal()

    # Read In Full Data
    data = pd.read_parquet('/Users/jerryinyang/Code/quantbt/data/prices/BTCUSDT.parquet')
    data.columns = data.columns.str.lower()

    x = np.log(data['close'].to_numpy())
    # x = x[-100:]

    miner = PatternMiner(6, 24, 10)
    print(miner.train(x))

    # pd.Series(x).plot()
    # for i in range(len(pips_x)):
    #     plt.plot(pips_x[i], pips_y[i], marker='o', color='red')

    # plt.show()