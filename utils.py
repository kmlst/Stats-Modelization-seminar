import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import cv2

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
    
    def du_shapley_value(self):
        """ Calculate DU-Shapley values for initialized players. """
        psi = np.zeros(self.I)
        pooled_data = np.concatenate(self.players_data) # data they would have if they pooled
        for i in range(self.I):
            others = [j for j in range(self.I) if j != i]
            mu_minus_i = sum(len(self.players_data[j]) for j in others) / (self.I - 1) # average number of data points each other player has

            n_i = len(self.players_data[i])
            u_with = 0
            u_without = 0
            for k in range(self.I - 1):
                model_with_i = self.model(max_iter=self.max_iter)
                model_without_i = self.model(max_iter=self.max_iter)
                # select a subset of pooled data
                pooled_indices = np.random.choice(pooled_data, int(k * mu_minus_i) + n_i, replace=False)
                model_with_i.fit(self.X_train[pooled_indices], self.y_train[pooled_indices])
                if k:
                    model_without_i.fit(self.X_train[pooled_indices[:int(k * mu_minus_i)]], self.y_train[pooled_indices[:int(k * mu_minus_i)]])
                    u_with = self.metric(self.y_test, model_with_i.predict(self.X_test))
                    u_without = self.metric(self.y_test, model_without_i.predict(self.X_test))
                else:
                    u_with += self.metric(self.y_test, model_with_i.predict(self.X_test))
            psi[i] = (u_with - u_without) / self.I
        return psi

    def plot_shapley_values(self, shapley_values):
        """ Visualization of Shapley values. """
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, self.I + 1), shapley_values, color='skyblue')
        plt.xlabel('Player')
        plt.ylabel('DU-Shapley Value')
        plt.title('DU-Shapley Values for Each Player')
        plt.xticks(range(1, self.I + 1))
        plt.show()

# Example usage
calculator = DU_mnist_ShapleyCalculator()
calculator.normal_players()
shapley_values = calculator.du_shapley_value()
calculator.plot_shapley_values(shapley_values)


