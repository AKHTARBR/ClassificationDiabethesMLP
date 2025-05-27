import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Fonctions d'activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes)
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            assert w.shape == (layer_sizes[i], layer_sizes[i+1])
            assert b.shape == (1, layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == self.layer_sizes[0]

        self.activations = [X]
        self.z_values = []

        A = X
        for i in range(len(self.weights) - 1):
            z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            A = relu(z)
            self.activations.append(A)

        z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = sigmoid(z)
        self.activations.append(output)

        return output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-15  # Pour éviter log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_accuracy(self, y_true, y_pred):
        preds = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(preds == y_true)
        return accuracy

    def backward(self, X, y, outputs):
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        dZ = outputs - y  # Dernière couche
        self.d_weights[-1] = self.activations[-2].T @ dZ / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dA = dZ @ self.weights[i + 1].T
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = self.activations[i].T @ dZ / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                outputs = self.forward(X_batch)
                self.backward(X_batch, y_batch, outputs)

            y_train_pred = self.forward(X)
            y_val_pred = self.forward(X_val)

            train_loss = self.compute_loss(y, y_train_pred)
            val_loss = self.compute_loss(y_val, y_val_pred)

            train_acc = self.compute_accuracy(y, y_train_pred)
            val_acc = self.compute_accuracy(y_val, y_val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        output = self.forward(X)
        predictions = (output >= 0.5).astype(int)
        return predictions

# Chargement des données
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values.reshape(-1, 1)

# Standardisation
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split (80% train+val, 20% test), puis train/val
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Initialisation et entraînement du réseau
nn = NeuralNetwork([X.shape[1], 16, 8, 1], learning_rate=0.01)
train_losses, val_losses, train_accuracies, val_accuracies = nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Évaluation
y_pred = nn.predict(X_test)
print("\nRapport de classification (Test set) :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

# Courbes de perte
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Courbe de perte")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

# Courbes d'accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Courbe d'accuracy")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
