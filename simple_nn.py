import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """
    Load and preprocess features, labels, and binary labels.
    
    Returns:
        train_X, val_X, y_train, y_val, train_bl, test_bl
    """
    def load_from_file(file_name):
        return np.loadtxt(file_name)

    features = load_from_file('features.txt')
    labels = load_from_file('labels.txt')
    binary_labels = load_from_file('binary_labels.txt')

    features_scaled = preprocessing.scale(features)

    train_X, val_X, train_y, test_y, train_bl, test_bl = train_test_split(
        features_scaled, labels, binary_labels, test_size=0.3, random_state=99
    )

    y_train = to_categorical(train_y)
    y_val = to_categorical(test_y)

    return train_X, val_X, y_train, y_val, train_bl, test_bl

def create_nn_model(n_cols):
    """
    Create a neural network model.
    
    Args:
        n_cols: Number of columns in the input data.
        
    Returns:
        model: Compiled Keras model.
    """
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(n_cols,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

    return model

def train_model(model, train_X, y_train, val_X, y_val, learning_rate=0.01, epochs=10, batch_size=100):
    """
    Train the neural network model.
    
    Args:
        model: Compiled Keras model.
        train_X: Training features.
        y_train: Training labels.
        val_X: Validation features.
        y_val: Validation labels.
        learning_rate: Learning rate for the optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Returns:
        model: Trained Keras model.
        history: Training history.
    """
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
	
	log_dir = "C:\\logs\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1, 
                                                      write_graph=True, 
                                                      write_images=True,
                                                      update_freq='epoch',
                                                      profile_batch=2,
                                                      embeddings_freq=0,
                                                      embeddings_metadata=None)

	history = model.fit(
    train_X, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_X, y_val),
    callbacks=[tensorboard_callback]
	)

	
    return model, history

def evaluate_model(model, test_data, y_test_data):
    """
    Evaluate the neural network model.
    
    Args:
        model: Trained Keras model.
        test_data: Test features.
        y_test_data: Test labels.
    """
    loss, accuracy = model.evaluate(test_data, y_test_data)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_percent, cmap='coolwarm')

    ax.set_xticks(np.arange(len(np.unique(y_true))))
    ax.set_yticks(np.arange(len(np.unique(y_true))))
    ax.set_xticklabels(['Predicted: {}'.format(i) for i in np.unique(y_true)], fontweight='bold', fontsize=14)
    ax.set_yticklabels(['True: {}'.format(i) for i in np.unique(y_true)], fontweight='bold', fontsize=14)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=16)

    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            text = '{:d}\n({:.2%})'.format(cm[i, j], cm_percent[i, j])
            ax.text(j, i, text, ha='center', va='center', color='w', fontweight='bold', fontsize=14)

    plt.show()

# Main execution
train_X, val_X, y_train, y_val, test_data, y_test_data = load_and_preprocess_data()
n_cols = train_X.shape[1]
n_rows = train_X.shape[0]

model = create_nn_model(n_cols)
model, history = train_model(model, train_X, y_train, val_X, y_val)
evaluate_model(model, test_data, y_test_data)

y_train_pred = np.argmax(model.predict(train_X), axis=-1)
plot_confusion_matrix(np.argmax(y_train, axis=1), y_train_pred)

y_val_pred = np.argmax(model.predict(val_X), axis=-1)
plot_confusion_matrix(np.argmax(y_val, axis=1), y_val_pred)

y_test_pred = np.argmax(model.predict(test_data), axis=-1)
plot_confusion_matrix(np.argmax(y_test_data, axis=1), y_test_pred)
