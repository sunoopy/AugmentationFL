import tensorflow as tf
import numpy as np

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

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    num_clients = 5
    num_samples = 100  # Per client
    num_classes = 10
    
    # Initialize global model and client models
    global_model = SimpleModel()
    client_models = [SimpleModel() for _ in range(num_clients)]
    
    # Generate synthetic data using ZSDG for each client
    client_data_list = []
    client_labels_list = []
    for i in range(num_clients):
        data, labels = zsdg(client_models[i], num_samples, num_classes)
        client_data_list.append(data)
        client_labels_list.append(labels)
    
    # Concatenate client data and labels
    client_data = tf.concat(client_data_list, axis=0)
    client_labels = tf.concat(client_labels_list, axis=0)
    
    # Perform FedAvg
    global_model = fedavg(client_models, global_model, client_data, client_labels, epochs=5)
    
    print("Federated learning with ZSDG completed.")
