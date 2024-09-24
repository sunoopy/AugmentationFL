import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

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