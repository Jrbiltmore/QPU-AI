
# /AI_ML_Modules/deep_learning_models.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, Bidirectional

def build_advanced_rnn_model(input_dim, output_dim, units, dropout_rate):
    # Advanced Recurrent Neural Network for handling sequence data
    inputs = Input(shape=(None, input_dim))
    x = Embedding(input_dim=input_dim, output_dim=256, mask_zero=True)(inputs)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units))(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example of building a model
if __name__ == '__main__':
    model = build_advanced_rnn_model(input_dim=1000, output_dim=100, units=200, dropout_rate=0.5)
    model.summary()
