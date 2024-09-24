import tensorflow as tf
import numpy as np
from keras.datasets import mnist

# Define a simple model with Batch Normalization using TensorFlow
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(256, input_shape=(784,))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(128)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = tf.nn.relu(self.bn1(self.fc1(inputs), training=training))
        x = tf.nn.relu(self.bn2(self.fc2(x), training=training))
        return self.fc3(x)

# Zero-Shot Data Generation (ZSDG) Algorithm
def zsdg(model, num_samples, num_classes):
    # Get batch normalization statistics from the model
    mu_list = []
    sigma_list = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            mu_list.append(layer.moving_mean)
            sigma_list.append(tf.sqrt(layer.moving_variance))
    
    # Generate synthetic data
    fake_data = []
    fake_labels = []
    for _ in range(num_samples):
        # Generate synthetic input data from Gaussian distribution
        x_synthetic = []
        for mu, sigma in zip(mu_list, sigma_list):
            x_synthetic.append(tf.random.normal(shape=mu.shape, mean=mu, stddev=sigma))
        x_synthetic = tf.concat(x_synthetic, axis=0)  # Concatenate the layers' outputs
        
        # Randomly assign labels
        y_synthetic = tf.random.uniform(shape=(1,), minval=0, maxval=num_classes, dtype=tf.int32)
        
        fake_data.append(x_synthetic)
        fake_labels.append(y_synthetic)
    
    fake_data = tf.stack(fake_data)
    fake_labels = tf.stack(fake_labels)
    
    return fake_data, fake_labels

# FedAvg Algorithm for Federated Learning
def fedavg(models, global_model, client_data, client_labels, epochs=1):
    client_weights = [model.get_weights() for model in models]
    
    # Averaging model weights
    avg_weights = []
    for weights in zip(*client_weights):
        avg_weights.append([np.mean(w, axis=0) for w in zip(*weights)])
    
    # Update the global model with the averaged weights
    global_model.set_weights(avg_weights)
    
    # Compile the model
    global_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    
    # Train global model on the aggregated data (FedAvg concept)
    global_model.fit(client_data, client_labels, epochs=epochs)
    
    return global_model

# Hierarchical Federated Learning (HFL)
def hierarchical_fedavg(global_model, regional_models, clients_per_region, num_regions, num_clients, num_samples, num_classes, epochs=1):
    # Step 1: Each client generates data using ZSDG and trains their local model
    regional_weights = []
    
    for i in range(num_regions):
        client_models = [SimpleModel() for _ in range(clients_per_region)]
        client_data_list = []
        client_labels_list = []
        
        # Generate synthetic data using ZSDG for each client
        for j in range(clients_per_region):
            data, labels = zsdg(client_models[j], num_samples, num_classes)
            client_data_list.append(data)
            client_labels_list.append(labels)
        
        # Concatenate client data and labels within each region
        client_data = tf.concat(client_data_list, axis=0)
        client_labels = tf.concat(client_labels_list, axis=0)
        
        # Regional server aggregates the clients' models (FedAvg within region)
        regional_model = fedavg(client_models, regional_models[i], client_data, client_labels, epochs)
        regional_weights.append(regional_model.get_weights())
    
    # Step 2: Global server aggregates the models from regional servers (FedAvg across regions)
    avg_weights = []
    for weights in zip(*regional_weights):
        avg_weights.append([np.mean(w, axis=0) for w in zip(*weights)])
    
    global_model.set_weights(avg_weights)
    
    return global_model

# Testing the global model on MNIST dataset
def test_global_model(global_model):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 784) / 255.0
    
    # Compile and evaluate the global model on the MNIST test set
    global_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    
    test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy on MNIST: {test_acc}")

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    num_regions = 3
    clients_per_region = 5
    num_clients = num_regions * clients_per_region
    num_samples = 100  # Per client
    num_classes = 10
    epochs = 5
    
    # Initialize global model and regional models
    global_model = SimpleModel()
    regional_models = [SimpleModel() for _ in range(num_regions)]
    
    # Perform Hierarchical FedAvg
    global_model = hierarchical_fedavg(global_model, regional_models, clients_per_region, num_regions, num_clients, num_samples, num_classes, epochs)
    
    # Test the global model on MNIST dataset
    test_global_model(global_model)
