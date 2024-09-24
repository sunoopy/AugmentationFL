
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class ZSDGFedAvgHFL:
    def __init__(self, num_clients, model_fn, num_rounds, local_epochs):
        self.num_clients = num_clients
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.global_model = self.model_fn()

    def zsdg(self, model, num_samples):
        # Get batch normalization layers
        bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
        
        # fake data generation
        fake_data = tf.Variable(tf.random.normal(shape=(num_samples,) + model.input_shape[1:]))
        fake_labels = tf.random.uniform(shape=(num_samples,), minval=0, maxval=model.output_shape[-1], dtype=tf.int32)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        
        for _ in range(self.local_epochs):
            with tf.GradientTape() as tape:
                # Forward propagation
                activations = [fake_data]
                for layer in model.layers:
                    activations.append(layer(activations[-1]))
                
                # Gather BN statistics
                bn_stats = []
                for layer in bn_layers:
                    bn_stats.append((layer.moving_mean, layer.moving_variance))
                
                # Compute loss (placeholder - replace with actual loss calculation)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(fake_labels, activations[-1]))
            
            # Backward propagation
            grads = tape.gradient(loss, [fake_data])
            optimizer.apply_gradients(zip(grads, [fake_data]))
        
        return fake_data.value(), fake_labels

    def client_update(self, client_model, fake_data, fake_labels):
        for _ in range(self.local_epochs):
            client_model.fit(fake_data, fake_labels, epochs=1, verbose=0)
        return client_model.get_weights()

    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for client_w in client_weights:
            for i, w in enumerate(client_w):
                avg_weights[i] += w / self.num_clients
        return avg_weights

    def train(self):
        for round in range(self.num_rounds):
            print(f"Round {round + 1}/{self.num_rounds}")
            
            client_weights = []
            for i in range(self.num_clients):
                client_model = self.model_fn()
                client_model.set_weights(self.global_model.get_weights())
                
                fake_data, fake_labels = self.zsdg(client_model, num_samples=100)  # Adjust num_samples as needed
                
                updated_weights = self.client_update(client_model, fake_data, fake_labels)
                client_weights.append(updated_weights)
            
            avg_weights = self.aggregate_weights(client_weights)

            self.global_model.set_weights(avg_weights)
        
        return self.global_model

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_non_iid_mnist_data(num_clients=100, samples_per_client=100, max_labels_per_client=2):

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_train, y_train = shuffle(x_train, y_train)

    # Create non-IID distribution
    client_data = []
    available_labels = list(range(10))
    
    for i in range(num_clients):
        if len(available_labels) < max_labels_per_client:
            available_labels = list(range(10))
        
        client_labels = np.random.choice(available_labels, max_labels_per_client, replace=False)
        for label in client_labels:
            available_labels.remove(label)
        
        client_indices = np.where(np.isin(y_train, client_labels))[0][:samples_per_client]
        client_x = x_train[client_indices]
        client_y = y_train[client_indices]
        
        client_data.append((client_x, client_y))
    
    return client_data

def test_model(model, test_data):
    x_test, y_test = test_data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy

# Main execution
if __name__ == "__main__":

    client_data = create_non_iid_mnist_data()

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    test_data = (x_test[-10000:], y_test[-10000:])

    # Initialize federated learning
    federated_learning = ZSDGFedAvgHFL(num_clients=100, model_fn=create_model, num_rounds=50, local_epochs=5)

    final_model = federated_learning.train()

    test_loss, test_accuracy = test_model(final_model, test_data)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

    for i, (client_x, client_y) in enumerate(client_data[:5]):  # Test on first 5 clients
        client_loss, client_accuracy = test_model(final_model, (client_x, client_y))
        print(f"Client {i+1} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.4f}")