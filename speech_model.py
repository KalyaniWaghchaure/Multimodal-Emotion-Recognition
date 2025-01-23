
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

import os

dataset_path = 'C:\\Users\\kalya\\.cache\\kagglehub\\datasets\\ejlok1\\toronto-emotional-speech-set-tess\\versions\\1'

paths = []
labels = []

for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1].split('.')[0]
        labels.append(label.lower())

        if len(paths) == 2800:
            break

print("Dataset is Loaded")


len(paths)

paths[:5]

labels[:5]


## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

df['label'].value_counts()
sns.countplot(df['label'])

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    # Replaced librosa.display.waveplot with librosa.display.waveshow
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectogram(data, sr, emotion):
     x = librosa.stft(data)
     xdb = librosa.amplitude_to_db(abs(x))
     plt.figure(figsize=(11,4))
     plt.title(emotion, size=20)
     librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
     plt.colorbar()

def extract_mfcc(filename):
     y, sr = librosa.load(filename, duration=3, offset=0.5)
     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
     return mfcc

extract_mfcc(df['speech'][0])

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc

X = [x for x in X_mfcc]
X = np.array(X)
X.shape

## input split
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])

y = y.toarray()
y.shape

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, GaussianNoise
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.optimizers import Adam

# Add noise to input data for better regularization
X_noisy = X + np.random.normal(0, 0.01, X.shape)

# Create model with stronger regularization
model = Sequential([
    # Add noise layer
    GaussianNoise(0.01, input_shape=(40,1)),

    # First LSTM layer with strong regularization
    LSTM(64, return_sequences=True,
         kernel_regularizer=l2(0.02),
         recurrent_regularizer=l2(0.02),
         activity_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.5),

    # Second LSTM layer
    LSTM(32, return_sequences=False,
         kernel_regularizer=l2(0.02),
         recurrent_regularizer=l2(0.02),
         activity_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.5),

    # Dense layer with strong regularization
    Dense(16, activation='relu',
          kernel_regularizer=l2(0.02),
          activity_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(7, activation='softmax',
          kernel_regularizer=l2(0.02))
])

# Compile model with reduced learning rate
model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.0005),
             metrics=['accuracy'])

model.summary()

# Add callbacks with stricter early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train the model with more aggressive regularization
history = model.fit(
    X_noisy, y,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    shuffle=True,
    verbose=1
)


# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final metrics
print(f"Final Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

# Print min validation loss and its corresponding accuracy
min_val_loss_idx = np.argmin(history.history['val_loss'])
print(f"\nBest model performance at epoch {min_val_loss_idx + 1}:")
print(f"Training Accuracy: {history.history['accuracy'][min_val_loss_idx] * 100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][min_val_loss_idx] * 100:.2f}%")

# save_model.py
import pickle
import joblib
import numpy as np
from keras.models import load_model

# Save the model
model.save('speech_emotion_model.h5')

# Save the OneHotEncoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(enc, f)


