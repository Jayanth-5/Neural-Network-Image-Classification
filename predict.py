import tensorflow as tf
import numpy as np
import sys

# Load class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model = tf.keras.models.load_model('cifar10_model.h5')

def predict_image(image_path):
    # Load and preprocess the image (assuming 32x32x3 input)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predicted_class, confidence = predict_image(image_path)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
