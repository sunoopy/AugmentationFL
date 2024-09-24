import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class FedZDAC:
    def __init__(self, num_clients, model_fn, num_rounds, local_epochs, client_fraction=0.5):
        self.num_clients = num_clients
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.global_model = self.model_fn()
        self.client_fraction = client_fraction

    def zsdg(self, model, num_samples):
        # ZSDG implementation (same as before)
        num_classes = model.output_shape[-1]
        samples_per_class = num_samples // num_classes

        fake_data_list = []
        fake_labels_list = []

        for class_label in range(num_classes):
            class_data = tf.random.normal(shape=(samples_per_class,) + model.input_shape[1:])
            class_labels = tf.ones(shape=(samples_per_class,), dtype=tf.int32) * class_label
            fake_data_list.append(class_data)
            fake_labels_list.append(class_labels)

        fake_data = tf.concat(fake_data_list, axis=0)
        fake_labels = tf.concat(fake_labels_list, axis=0)

        fake_data = tf.Variable(fake_data)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        for _ in range(self.local_epochs):
            with tf.GradientTape() as tape:
                predictions = model(fake_data, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(fake_labels, predictions))
            
            grads = tape.gradient(loss, [fake_data])
            optimizer.apply_gradients(zip(grads, [fake_data]))

        return fake_data.numpy(), fake_labels.numpy()

    def client_update(self, client_model, real_data, fake_data):
        x_real, y_real = real_data
        x_fake, y_fake = fake_data
        
        # Combine real and fake data
        x_combined = np.concatenate([x_real, x_fake], axis=0)
        y_combined = np.concatenate([y_real, y_fake], axis=0)
        
        # Shuffle combined data
        indices = np.arange(len(x_combined))
        np.random.shuffle(indices)
        x_combined = x_combined[indices]
        y_combined = y_combined[indices]

        client_model.fit(x_combined, y_combined, epochs=self.local_epochs, verbose=0)
        return client_model.get_weights()

    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for client_w in client_weights:
            for i, w in enumerate(client_w):
                avg_weights[i] += w / len(client_weights)
        return avg_weights

    def train(self, client_data):
        for round in range(self.num_rounds):
            print(f"Round {round + 1}/{self.num_rounds}")
            
            # Randomly select subset of clients
            num_selected = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(self.num_clients, num_selected, replace=False)
            
            client_weights = []
            for i in selected_clients:
                client_x, client_y = client_data[i]
                client_model = self.model_fn()
                client_model.set_weights(self.global_model.get_weights())

                # Generate fake data using ZSDG
                fake_data, fake_labels = self.zsdg(client_model, num_samples=len(client_x))
                
                # Update client model with mix of real and fake data
                updated_weights = self.client_update(client_model, (client_x, client_y), (fake_data, fake_labels))
                client_weights.append(updated_weights)
            
            # Server aggregates client models
            self.global_model.set_weights(self.aggregate_weights(client_weights))

        return self.global_model

# Helper functions (create_model, create_mnist_data, test_model) remain the same

# Main function
if __name__ == "__main__":
    num_clients = 10
    num_rounds = 5
    local_epochs = 1
    client_fraction = 0.5  # Select 50% of clients each round

    client_data = create_mnist_data(num_clients=num_clients, iid=False)
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    test_data = (x_test[-10000:], y_test[-10000:])

    fed_zdac = FedZDAC(num_clients=num_clients, model_fn=create_model, 
                       num_rounds=num_rounds, local_epochs=local_epochs,
                       client_fraction=client_fraction)

    final_model = fed_zdac.train(client_data)

    test_loss, test_accuracy = test_model(final_model, test_data)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # Evaluate on client data
    for i, (client_x, client_y) in enumerate(client_data[:5]):  # Test on first 5 clients
        client_loss, client_accuracy = test_model(final_model, (client_x, client_y))
        print(f"Client {i+1} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.4f}")