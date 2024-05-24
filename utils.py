import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import cv2
import itertools

class DU_mnist_ShapleyCalculator:
    def __init__(self, dataset='mnist_784', model=LogisticRegression, metric=accuracy_score, max_iter=1000):
        self.dataset = dataset
        self.model = model
        self.metric = metric
        self.max_iter = max_iter
        self.load_data_normal()

    def load_data_normal(self):
        """ Load the MNIST dataset and split it into training and testing sets."""
        mnist = fetch_openml(self.dataset)
        X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=39)
    
    def normal_players(self, I=10, mean=1000, std=150):
        """
        Initializes player data by randomly selecting unique subsets of indices from training data. Each subset corresponds to a player, with the subset size normally distributed.
        Each player has unique data points.

        Parameters:
        - I (int): Number of players. Defaults to 10.
        - mean (int): Mean of the normal distribution. Defaults to 1000.
        - std (int): Standard deviation of the normal distribution. Defaults to 150.
        Attributes updated:
        - self.I (int): Stores the number of players.
        - self.players_data (list of lists): Each sublist contains the unique indices from self.X_train assigned to a player.
        """
        self.I = I
        players_data = []
        available_indices = set(range(len(self.X_train))) - set(range(1000))
        for i in range(I):
            n = int(np.random.normal(mean, std))
            player_indices = np.random.choice(list(available_indices), n, replace=False)
            players_data.append(player_indices)
            available_indices -= set(player_indices) # each player has unique data points
        self.players_data = players_data

    def degrade_data(self, player_indices, degradation_type='blur', severity=1):
        """Degrades MNIST image data for given players.

        Parameters:
        - player_indices (list): List of player indices whose data needs degradation.
        - degradation_type (str): Type of degradation. Options: 'blur', 'noise'. Defaults to 'blur'.
        - severity (int): Level of degradation severity (1-5). Defaults to 1.
        """
        if degradation_type == 'blur':
            kernel_size = 2 * severity + 1  # Kernel size increases with severity
            for i in player_indices:
                for j in self.players_data[i]:
                    img = self.X_train[j].reshape(28, 28)
                    self.X_train[j] = cv2.blur(img, (kernel_size, kernel_size)).flatten()
        elif degradation_type == 'noise':
            noise_scale = severity * 0.1  # Noise scale increases with severity
            for i in player_indices:
                for j in self.players_data[i]:
                    noise = np.random.normal(0, noise_scale, size=self.X_train[j].shape)
                    self.X_train[j] = np.clip(self.X_train[j] + noise, 0, 1)
    
    def du_shapley_value(self):
        """ Calculate DU-Shapley values for initialized players. """
        psi = np.zeros(self.I)
        pooled_data = np.concatenate(self.players_data) 
        for i in range(self.I):
            others = [j for j in range(self.I) if j != i]
            mu_minus_i = sum(len(self.players_data[j]) for j in others) / (self.I - 1) 

            n_i = len(self.players_data[i])
            
            model_with_i = self.model(max_iter=self.max_iter)
            model_with_i.fit(self.X_train[self.players_data[i]], self.y_train[self.players_data[i]])
            u_with = self.metric(self.y_test, model_with_i.predict(self.X_test))
            psi[i] = u_with / self.I

            for k in range(1, self.I - 1): 
                u_with = 0  
                model_with_i = self.model(max_iter=self.max_iter)
                model_without_i = self.model(max_iter=self.max_iter)
                pooled_indices = np.random.choice(pooled_data, int(k * mu_minus_i) + n_i, replace=False)
                model_with_i.fit(self.X_train[pooled_indices], self.y_train[pooled_indices])
                u_with = self.metric(self.y_test, model_with_i.predict(self.X_test))

                model_without_i.fit(self.X_train[pooled_indices[:int(k * mu_minus_i)]], self.y_train[pooled_indices[:int(k * mu_minus_i)]])
                u_without = self.metric(self.y_test, model_without_i.predict(self.X_test)) 
                
                psi[i] += (u_with - u_without) / self.I

        return psi

    def calculate_true_shapley(self):
        """Calculate the true Shapley values for the players."""
        true_shapley = np.zeros(self.I)
        all_players = set(range(self.I))
        for i in range(self.I):
            for coalition_size in range(self.I):
                for coalition in itertools.combinations(all_players - {i}, coalition_size):
                    coalition = set(coalition)
                    coalition_with_i = coalition | {i}
                    if coalition: 
                        true_shapley[i] += (self.evaluate_coalition(coalition_with_i) - self.evaluate_coalition(coalition)) / (self.I * np.math.comb(self.I-1, coalition_size))
                    else:
                        true_shapley[i] += self.evaluate_coalition(coalition_with_i) / (self.I * np.math.comb(self.I-1, coalition_size)) 
        return true_shapley

    def evaluate_coalition(self, coalition):
        """Train and evaluate a model on data from a given coalition of players.
        
        Parameters:
        - coalition (set): Set of player indices forming the coalition.
        """
        coalition_indices = np.concatenate([self.players_data[i] for i in coalition])
        model = self.model(max_iter=self.max_iter)
        model.fit(self.X_train[coalition_indices], self.y_train[coalition_indices])
        return self.metric(self.y_test, model.predict(self.X_test))

    def plot_shapley_values(self, shapley_values):
        """ Visualization of Shapley values. """
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, self.I + 1), shapley_values, color='skyblue')
        plt.xlabel('Player')
        plt.ylabel('DU-Shapley Value')
        plt.title('DU-Shapley Values for Each Player')
        plt.xticks(range(1, self.I + 1))
        plt.show()

