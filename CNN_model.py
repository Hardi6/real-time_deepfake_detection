import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
import matplotlib.pyplot as plt

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def label_images(root_dir):
    data = []
    for folder in ['Train', 'Test', 'Validation']:
        for label in ['real', 'fake']:
            folder_path = os.path.join(root_dir, folder, label)
            if not os.path.exists(folder_path):
                print(f"Folder does not exist: {folder_path}")
                continue
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)
                    image_label = 0 if label == 'real' else 1
                    if detect_faces(image_path):
                        data.append([image_path, image_label])
    return data


# Define the root directory
root_dir = "D://Facemesh//Videosdetect//MiniData"

# Label images and prepare data
data = label_images(root_dir)

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['ImagePath', 'Label'])

# Save the DataFrame to a CSV file
df.to_csv('Mini.csv', index=False)


# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    return hog_features.flatten()  # Flatten to 1D array


# Function to extract features from images specified in a CSV file
def extract_features_from_csv(csv_file, sequence_length):
    # Read CSV file
    df = pd.read_csv(csv_file)
    # Initialize lists to store image paths and features
    image_paths = df['ImagePath'].tolist()
    features = []
    # Iterate over each image path
    for path in image_paths:
        # Load image
        image = cv2.imread(path)
        if image is not None:  # Check if image loading is successful
            # Extract HOG features
            hog_features = extract_hog_features(image)
            features.append(hog_features)
        else:
            print("Failed to load image:", path)

    # Pad or truncate sequences to ensure they all have the same length
    padded_features = []
    for feat in features:
        if len(feat) < sequence_length:
            # Pad with zeros if sequence is shorter than the desired length
            padded_feat = np.pad(feat, (0, sequence_length - len(feat)), mode='constant')
        else:
            # Truncate if sequence is longer than the desired length
            padded_feat = feat[:sequence_length]
        padded_features.append(padded_feat)

    return np.array(padded_features)  # Convert features to NumPy array


# CSV file containing image paths
csv_file = 'Mini.csv'

# Extract features from images
sequence_length = 100  # Define sequence length
extracted_features = extract_features_from_csv(csv_file, sequence_length)

# Save extracted features to a NumPy file
np.save('extracted_features_sequences.npy', extracted_features)
extracted_features = np.load('extracted_features_sequences.npy')

# Load the saved extracted features
from tensorflow.keras.layers import Dense, SimpleRNN

extracted_features = np.load('extracted_features_sequences.npy')

# Load labels from the CSV file
df = pd.read_csv('MiniDataLabels.csv')
labels = df['Label'].values

# Ensure the number of samples in features and labels match
if len(extracted_features) > len(labels):
    extracted_features = extracted_features[:len(labels)]
elif len(extracted_features) < len(labels):
    labels = labels[:len(extracted_features)]

# Now, the number of samples in features and labels should match
assert len(extracted_features) == len(labels), "Number of samples in features and labels should match"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(extracted_features, labels, test_size=0.2, random_state=42)

# Reshape features to include the number of features per sample
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the RNN model with dropout
model = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.5),  # Add dropout with a rate of 0.5
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Function to preprocess the input image
def preprocess_image(image_path, sequence_length):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)

    # Flatten HOG features and pad/truncate to match sequence length
    if len(hog_features) < sequence_length:
        padded_features = np.pad(hog_features, ((0, sequence_length - len(hog_features)), (0, 0)), mode='constant')
    else:
        padded_features = hog_features[:sequence_length]

    # Reshape to match the input shape of the model
    preprocessed_image = padded_features.reshape(1, sequence_length, -1)

    return preprocessed_image


# Function to predict the label for the input image
def predict_image_label(image_path, model, sequence_length):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path, sequence_length)

    if preprocessed_image is not None:
        # Make predictions
        prediction = model.predict(preprocessed_image)

        # Convert prediction to a human-readable label (real or fake)
        label = "real" if prediction < 0.5 else "fake"

        return label
    else:
        return "Preprocessing failed due to image loading error."


# Example usage:
input_image_path = "D://Facemesh//Videosdetect//Dataset//Validation//Fake//fake_12904.jpg"  # Replace with the path to your input image

# Predict label for the input image
predicted_label = predict_image_label(input_image_path, model, sequence_length)

print("Predicted Label:", predicted_label)