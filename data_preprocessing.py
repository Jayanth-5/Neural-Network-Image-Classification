import tensorflow as tf
from tensorflow.keras import datasets

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return (train_images, train_labels), (test_images, test_labels), class_names

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels), class_names = load_and_preprocess_data()
    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Class names: {class_names}")
