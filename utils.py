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

    def utility(self, n):
        """ Calculate utility for a player with n data points chosen at random"""
        model = self.model(max_iter=self.max_iter)
        model.fit(self.X_train[:n], self.y_train[:n])
        return self.metric(self.y_test, model.predict(self.X_test))
    
    def du_shapley_value(self):
        """ Calculate DU-Shapley values for initialized players. """
        psi = np.zeros(self.I)
        pooled_data = np.concatenate(self.players_data) 
        for i in range(self.I):
            others = [j for j in range(self.I) if j != i]
            mu_minus_i = sum(len(self.players_data[j]) for j in others) / (self.I - 1) 

            n_i = len(self.players_data[i])

            # Case k=0
            model_with_i = self.model(max_iter=self.max_iter)
            model_with_i.fit(self.X_train[self.players_data[i]], self.y_train[self.players_data[i]])
            u_with = self.metric(self.y_test, model_with_i.predict(self.X_test))
            psi[i] = u_with / self.I

            # Calculate contributions for k = 1 to I-2
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

# --- Fonctions pour la dégradation des données ---

def add_noise(data, std_dev):
    """Ajoute du bruit gaussien aux données."""
    noise = np.random.normal(0, std_dev, size=data.shape)
    return np.clip(data + noise, 0, 1)

def apply_blur(data, kernel_size):
    """Applique un flou gaussien aux données."""
    blurred_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        img = data[i].reshape(28, 28)
        blurred_data[i] = cv2.blur(img, (kernel_size, kernel_size)).flatten()
    return blurred_data

def occlude_data(data, occlusion_ratio):
    """Masque une partie des données avec des valeurs aléatoires."""
    mask = np.random.rand(*data.shape) < occlusion_ratio
    occluded_data = data.copy()
    occluded_data[mask] = np.random.rand(np.sum(mask))
    return occluded_data

def downsample_data(data, downsampling_factor):
    """Réduit la résolution des données."""
    downsampled_data = np.zeros((data.shape[0], 784 // downsampling_factor**2))
    for i in range(data.shape[0]):
        img = data[i].reshape(28, 28)
        downsampled_img = cv2.resize(img, (28 // downsampling_factor, 28 // downsampling_factor))
        downsampled_data[i] = downsampled_img.flatten()
    return downsampled_data

def mislabel_data(data, labels, mislabeling_ratio):
    """Introduit des erreurs dans les étiquettes des données."""
    mislabeled_labels = labels.copy()
    num_mislabeled = int(len(labels) * mislabeling_ratio)
    mislabeled_indices = np.random.choice(len(labels), num_mislabeled, replace=False)
    for i in mislabeled_indices:
        mislabeled_labels[i] = np.random.choice([l for l in range(10) if l != labels[i]])
    return data, mislabeled_labels

# --- Méthodologie pour vérifier l'impact de la qualité des données ---

def run_experiment(I, degradation_type, degradation_level, repetitions=10):
    """Exécute une expérience pour un scénario donné."""

    results = {
        "DU-Shapley": [],
        "True Shapley": [],
        "degradation_type": degradation_type,
        "degradation_level": degradation_level,
        "I": I
    }

    for _ in range(repetitions):
        calculator = DU_mnist_ShapleyCalculator()
        calculator.normal_players(I=I, mean=100, std=10)

        if degradation_type == "noise":
            calculator.X_train = add_noise(calculator.X_train, degradation_level)
        elif degradation_type == "blur":
            calculator.X_train = apply_blur(calculator.X_train, degradation_level)
        elif degradation_type == "occlusion":
            calculator.X_train = occlude_data(calculator.X_train, degradation_level)
        elif degradation_type == "downsampling":
            calculator.X_train = downsample_data(calculator.X_train, degradation_level)
        elif degradation_type == "mislabeling":
            calculator.X_train, calculator.y_train = mislabel_data(calculator.X_train, calculator.y_train, degradation_level)

        du_shapley = calculator.du_shapley_value()
        true_shapley = calculator.calculate_true_shapley()

        results["DU-Shapley"].append(du_shapley)
        results["True Shapley"].append(true_shapley)

    return results

