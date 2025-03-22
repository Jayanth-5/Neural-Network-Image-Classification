import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_and_train_model(train_images, train_labels, test_images, test_labels, epochs=20, batch_size=64):
    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels),
                        batch_size=batch_size)

    # Save model
    model.save('cifar10_model.h5')
    print("Model saved as cifar10_model.h5")

    return model, history

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    (train_images, train_labels), (test_images, test_labels), _ = load_and_preprocess_data()
    model, history = build_and_train_model(train_images, train_labels, test_images, test_labels)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
