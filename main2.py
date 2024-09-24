import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class ZSDGFedAvgHFL:
    def __init__(self, num_clients, model_fn, num_rounds, local_epochs, use_zsdg=True):
        self.num_clients = num_clients
        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.global_model = self.model_fn()
        self.use_zsdg = use_zsdg

    def zsdg(self, model, num_samples):
        if not self.use_zsdg:
            return None, None

        # ZSDG implementation with balanced labels
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

        print(f"Generated fake data: {fake_data.shape[0]} samples, {samples_per_class} per class")
        return fake_data.numpy(), fake_labels.numpy()

    def client_update(self, client_model, client_data, fake_data=None, fake_labels=None):
        if self.use_zsdg and fake_data is not None and fake_labels is not None:
            train_data = fake_data
            train_labels = fake_labels
        else:
            train_data, train_labels = client_data

        for _ in range(self.local_epochs):
            client_model.fit(train_data, train_labels, epochs=1, verbose=0)
        return client_model.get_weights()

    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for client_w in client_weights:
            for i, w in enumerate(client_w):
                avg_weights[i] += w / self.num_clients
        return avg_weights

    def evaluate_model(self, test_data):
        loss, accuracy = self.global_model.evaluate(test_data[0], test_data[1], verbose=0)
        return loss, accuracy

    def train(self, client_data):
        for round in range(self.num_rounds):
            print(f"Round {round + 1}/{self.num_rounds}")
            
            client_weights = []
            for i, (client_x, client_y) in enumerate(client_data):
                client_model = self.model_fn()
                client_model.set_weights(self.global_model.get_weights())

                print(f"Client {i+1} original labels: {np.unique(client_y)}")

                if self.use_zsdg:
                    fake_data, fake_labels = self.zsdg(client_model, num_samples=100)
                    print(f"Client {i+1} generated fake data: {fake_data.shape[0]} samples, unique labels: {np.unique(fake_labels)}")
                else:
                    fake_data, fake_labels = None, None

                updated_weights = self.client_update(client_model, (client_x, client_y), fake_data, fake_labels)
                client_weights.append(updated_weights)
            
            avg_weights = self.aggregate_weights(client_weights)
            self.global_model.set_weights(avg_weights)

            test_loss, test_accuracy = self.evaluate_model(test_data)
            print(f"Round {round + 1} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
        
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

# Main function
if __name__ == "__main__":
    # Parameters
    num_clients = 10
    num_rounds = 5
    local_epochs = 1
    use_zsdg = True  # Set to False to turn off ZSDG
    iid_data = False  # Set to True for IID data distribution

    client_data = create_mnist_data(num_clients=num_clients, iid=iid_data)
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    test_data = (x_test[-10000:], y_test[-10000:])

    federated_learning = ZSDGFedAvgHFL(num_clients=num_clients, model_fn=create_model, 
                                       num_rounds=num_rounds, local_epochs=local_epochs, 
                                       use_zsdg=use_zsdg)

    final_model = federated_learning.train(client_data)

    test_loss, test_accuracy = test_model(final_model, test_data)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

    # Evaluate on client data
    for i, (client_x, client_y) in enumerate(client_data[:5]):  # Test on first 5 clients
        client_loss, client_accuracy = test_model(final_model, (client_x, client_y))
        print(f"Client {i+1} - Loss: {client_loss:.4f}, Accuracy: {client_accuracy:.4f}")