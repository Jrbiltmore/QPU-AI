import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Bidirectional
from typing import Tuple

def build_advanced_rnn_model(input_dim: int, output_dim: int, units: int, dropout_rate: float) -> Model:
    """
    Builds an advanced recurrent neural network model for sequence data processing.

    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output.
        units (int): The number of units/neurons in the LSTM layers.
        dropout_rate (float): The dropout rate for regularization.

    Returns:
        Model: A Keras Model object representing the built RNN model.

    Raises:
        RuntimeError: If an error occurs during model construction.
    """
    try:
        # Define model architecture
        inputs = Input(shape=(None, input_dim))
        x = Embedding(input_dim=input_dim, output_dim=256, mask_zero=True)(inputs)
        x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(units))(x)
        outputs = Dense(output_dim, activation='softmax')(x)

        # Compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        raise RuntimeError(f"Error building the advanced RNN model: {str(e)}")

def save_model(model: Model, filepath: str) -> None:
    """
    Saves the Keras model to a file.

    Args:
        model (Model): The Keras model to save.
        filepath (str): The path to save the model.

    Returns:
        None

    Raises:
        ValueError: If the model is not a Keras Model instance.
        OSError: If an error occurs while saving the model to the specified filepath.
    """
    if not isinstance(model, Model):
        raise ValueError("The provided model is not a Keras Model instance.")
    
    try:
        model.save(filepath)
        print(f"Model saved successfully to {filepath}")
    except OSError as e:
        raise OSError(f"Error saving the model: {str(e)}")

def load_model(filepath: str) -> Model:
    """
    Loads a Keras model from a file.

    Args:
        filepath (str): The path to the saved model file.

    Returns:
        Model: The loaded Keras model.

    Raises:
        FileNotFoundError: If the specified file is not found.
        ValueError: If the loaded object is not a Keras Model instance.
        OSError: If an error occurs while loading the model from the specified filepath.
    """
    try:
        model = tf.keras.models.load_model(filepath)
        if not isinstance(model, Model):
            raise ValueError("The loaded object is not a Keras Model instance.")
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {str(e)}")
    except OSError as e:
        raise OSError(f"Error loading the model: {str(e)}")

def evaluate_model(model: Model, x_test, y_test) -> Tuple[float, float]:
    """
    Evaluates the performance of the Keras model on test data.

    Args:
        model (Model): The Keras model to evaluate.
        x_test: The input test data.
        y_test: The target test labels.

    Returns:
        Tuple[float, float]: A tuple containing the loss and accuracy scores.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

# Example of building, saving, loading, and evaluating a model
if __name__ == '__main__':
    try:
        # Build and train the model
        model = build_advanced_rnn_model(input_dim=1000, output_dim=100, units=200, dropout_rate=0.5)
        # Assume x_train, y_train, x_test, and y_test are defined
        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        # Save the model
        save_path = "advanced_rnn_model.h5"
        save_model(model, save_path)

        # Load the model
        loaded_model = load_model(save_path)

        # Evaluate the model
        loss, accuracy = evaluate_model(loaded_model, x_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    except Exception as e:
        print(f"Error encountered while building, training, saving, loading, or evaluating the model: {str(e)}")
