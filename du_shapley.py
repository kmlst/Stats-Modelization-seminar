import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# # Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()

# Function to calculate DU-Shapley value
def du_shapley_value(X, y, X_test, y_test, players_data):
    I = len(players_data)
    utility_values = np.zeros(I)

    for i in range(I):
        ni = len(players_data[i])
        subset_utility = []

        for k in range(I):
            subsets = combinations([j for j in range(I) if j != i], k)
            for subset in subsets:
                subset_indices = [idx for j in subset for idx in players_data[j]]
                if subset_indices:  # Check if subset_indices is not empty
                    X_train = X[subset_indices]
                    y_train = y[subset_indices]
                    utility = calculate_utility(X_train, y_train, X_test, y_test)
                    subset_utility.append(utility)

        utility_values[i] = np.mean(subset_utility) if subset_utility else 0

    return utility_values

# Define players' data (index slices)
I = 10
X_test = X[:1000]
y_test = y[:1000]

players_data = []
mean = 1000
std = 150

# Create a pool of available indices excluding test indices
available_indices = set(range(70000)) - set(range(1000))

# Ensure unique data points for each player
for i in range(I):
    n = int(np.random.normal(mean, std))
    players_data.append(np.random.choice(range(1000 + i*10000, (i+1)*10000), n, replace=False))


# Calculate DU-Shapley values
du_shapley_vals = du_shapley_value(X, y, X_test, y_test, players_data)
for i, val in enumerate(du_shapley_vals):
    print(f"Player {i+1} had {len(players_data[i])} data points and the DU-Shapley value is: {val:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(range(1, I + 1), du_shapley_vals, color='skyblue')
plt.xlabel('Player')
plt.ylabel('DU-Shapley Value')
plt.title('DU-Shapley Values for Each Player')
plt.xticks(range(1, I + 1))
plt.show()