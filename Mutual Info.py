import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.feature_selection import mutual_info_classif

# Function to select and load the dataset
def load_dataset(dataset_name):
    if dataset_name == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset_name == "fashionmnist":
        return tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == "cifar10":
        return tf.keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        return tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError("Dataset not supported.")

# Existing CNN model for training
def create_existing_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ConvNet for evaluation
def create_convnet_for_evaluation(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# RL Agent class
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Function to save synthetic dataset to a specified location
def save_synthetic_dataset(X_synthetic, y_synthetic, dataset_name, path):
    dataset_dir = os.path.join(path, f'images_{dataset_name}')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for i in range(X_synthetic.shape[0]):
        img = (X_synthetic[i] * 255).astype(np.uint8)
        img_path = os.path.join(dataset_dir, f'image_{i}_{y_synthetic[i]}.png')
        tf.keras.preprocessing.image.save_img(img_path, img)

    print(f"Synthetic dataset saved at {dataset_dir}")

# Function to create synthetic dataset using RL agent
def create_synthetic_dataset(X_train, y_train, num_examples_per_class, num_epochs, dataset_name):
    num_classes = len(np.unique(y_train))
    num_examples = num_examples_per_class * num_classes
    X_synthetic = np.zeros((num_examples, *X_train.shape[1:]))
    y_synthetic = np.zeros((num_examples,), dtype=int)

    for class_label in range(num_classes):
        class_indices = np.where(y_train == class_label)[0]
        selected_indices = np.random.choice(class_indices, num_examples_per_class, replace=False)

        start_idx = class_label * num_examples_per_class
        end_idx = start_idx + num_examples_per_class
        X_synthetic[start_idx:end_idx] = X_train[selected_indices]
        y_synthetic[start_idx:end_idx] = y_train[selected_indices].flatten()

    # Reshape for the channel dimension if needed
    if dataset_name.lower() in ['mnist', 'fashionmnist']:
        X_synthetic = X_synthetic.reshape(-1, 28, 28, 1)  # Add channels dimension for grayscale images
    elif dataset_name in ['cifar10', 'cifar100']:
        X_synthetic = X_synthetic.reshape(-1, 32, 32, 3)  # Assuming CIFAR images are 32x32x3

    state_size = X_synthetic.shape[1] * X_synthetic.shape[2] * X_synthetic.shape[3]
    agent = RLAgent(state_size, num_classes)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    existing_model = create_existing_cnn_model(X_synthetic.shape[1:], num_classes)

    for epoch in range(num_epochs):
        existing_model.fit(datagen.flow(X_synthetic, y_synthetic, batch_size=32), epochs=10, verbose=0)
        y_synthetic_pred = np.argmax(existing_model.predict(X_synthetic), axis=1)

        for i in range(num_examples):
            state = X_synthetic[i]
            action = y_synthetic_pred[i]
            reward = 1 if y_synthetic[i] == action else -1
            next_state = state
            done = True
            agent.remember(state.flatten(), action, reward, next_state.flatten(), done)

        agent.replay(32)

        train_loss, train_acc = existing_model.evaluate(X_synthetic, y_synthetic, verbose=0)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}")

    return X_synthetic, y_synthetic

# Function to train and evaluate synthetic dataset on a ConvNet
def train_and_evaluate_convnet(X_synthetic, y_synthetic, num_classes, X_test, y_test):
    input_shape = X_synthetic.shape[1:]

    X_train_syn, X_val_syn, y_train_syn, y_val_syn = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

    convnet = create_convnet_for_evaluation(input_shape, num_classes)

    convnet.fit(X_train_syn, y_train_syn, epochs=10, batch_size=32, verbose=2, validation_data=(X_val_syn, y_val_syn))

    val_loss, val_acc = convnet.evaluate(X_val_syn, y_val_syn, verbose=2)
    print(f"\nFinal accuracy on synthetic dataset: {val_acc * 100:.4f}%")

    # Evaluate on original test set
    test_loss, test_acc = convnet.evaluate(X_test, y_test, verbose=2)
    print(f"Final Test accuracy on synthetic dataset: {test_acc:.4f}")

# Main program to run the code
if __name__ == "__main__":
    # Define number of epochs for each dataset
    epochs_dict = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 25,
        "cifar100": 25
    }

    # Ask user for input
    dataset_name = input("Enter dataset name (mnist, fashionmnist, cifar10, cifar100): ").lower()
    num_images_per_class = int(input("Enter number of images per class (1, 10, 50): "))
    save_path = input("Enter the path to save synthetic dataset: ")

    # Load the selected dataset
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset_name)

    # Mutual information-based data condensation
    x_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_flat = y_train.flatten()  # Ensure y is 1D
    mi = mutual_info_classif(x_train_flat, y_train_flat)

    top_indices = np.argsort(mi)[-1000:]  # Select top 1000 samples with highest MI
    x_condensed, y_condensed = X_train[top_indices], y_train[top_indices]

    # Create synthetic dataset
    X_synthetic, y_synthetic = create_synthetic_dataset(x_condensed, y_condensed, num_images_per_class, epochs_dict[dataset_name], dataset_name)

    # Save the synthetic dataset
    save_synthetic_dataset(X_synthetic, y_synthetic, dataset_name, save_path)

    # Train and evaluate on the synthetic dataset
    train_and_evaluate_convnet(X_synthetic, y_synthetic, len(np.unique(y_train)), X_test, y_test)
