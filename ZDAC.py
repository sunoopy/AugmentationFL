import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from collections import Counter

class FedZDAC:
    def __init__(self, num_clients, model_fn, num_rounds, local_epochs, client_fraction=0.5, use_zsdg=True):
        self.num_clients = num_clients
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.global_model = self.model_fn()
        self.client_fraction = client_fraction
        self.use_zsdg = use_zsdg

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

    def client_update(self, client_model, real_data, fake_data=None):
        x_real, y_real = real_data
        
        if fake_data is not None and self.use_zsdg:
            x_fake, y_fake = fake_data
            # Combine real and fake data
            x_combined = np.concatenate([x_real, x_fake], axis=0)
            y_combined = np.concatenate([y_real, y_fake], axis=0)
        else:
            x_combined, y_combined = x_real, y_real
        
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

                if self.use_zsdg:
                    # Generate fake data using ZSDG
                    fake_data, fake_labels = self.zsdg(client_model, num_samples=len(client_x))
                    # Update client model with mix of real and fake data
                    updated_weights = self.client_update(client_model, (client_x, client_y), (fake_data, fake_labels))
                else:
                    # Update client model with only real data
                    updated_weights = self.client_update(client_model, (client_x, client_y))
                
                client_weights.append(updated_weights)
            
            # Server aggregates client models
            self.global_model.set_weights(self.aggregate_weights(client_weights))

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

def create_mnist_data(num_clients=10, samples_per_client=100, iid=False, max_labels_per_client=2):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)

    x_train, y_train = shuffle(x_train, y_train)

    client_data = []
    
    if iid:
        # IID data distribution
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_x = x_train[start_idx:end_idx]
            client_y = y_train[start_idx:end_idx]
            client_data.append((client_x, client_y))
    else:
        # Non-IID data distribution
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

def print_label_distribution(client_data):
    for i, (_, client_y) in enumerate(client_data):
        label_counts = Counter(client_y)
        print(f"Client {i+1} label distribution:")
        for label in range(10):
            count = label_counts.get(label, 0)
            percentage = count / len(client_y) * 100
            print(f"  Label {label}: {count} ({percentage:.2f}%)")
        print()

# Main function
if __name__ == "__main__":
    num_clients = 10
    num_rounds = 5
    local_epochs = 1
    client_fraction = 0.5  # Select 50% of clients each round
    
    # Allow user to choose data distribution and ZSDG settings
    iid_choice = input("Choose data distribution (IID/non-IID): ").lower()
    iid = iid_choice == "iid"
    
    zsdg_choice = input("Use Zero-Shot Data Generation (yes/no): ").lower()
    use_zsdg = zsdg_choice == "yes"

    client_data = create_mnist_data(num_clients=num_clients, iid=iid)
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    test_data = (x_test[-10000:], y_test[-10000:])

    # Print label distribution for each client
    print("\nLabel distribution for each client:")
    print_label_distribution(client_data)

    fed_zdac = FedZDAC(num_clients=num_clients, model_fn=create_model, 
                       num_rounds=num_rounds, local_epochs=local_epochs,
                       client_fraction=client_fraction, use_zsdg=use_zsdg)

    final_model = fed_zdac.train(client_data)

    test_loss, test_accuracy = test_model(final_model, test_data)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # Evaluate on client data
    print("\nEvaluation on client data:")
    for i, (client_x, client_y) in enumerate(client_data[:5]):  # Test on first 5 clients
        client_loss, client_accuracy = test_model(final_model, (client_x, client_y))
        print(f"Client {i+1} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.4f}")