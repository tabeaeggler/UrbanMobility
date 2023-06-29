from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.layers import MultiHeadAttention, Flatten
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

#https://www.tensorflow.org/text/tutorials/transformer
#https://keras.io/examples/timeseries/timeseries_transformer_classification/


 # Borough-specific Multi-series LSTM model -> single LSTM model on multiple time series (one for each borough).
 # This is a Borough-specific Multi-series LSTM model designed to forecast bike sharing demand for each borough in London separately. By treating each borough's data as an individual time series, the model can learn unique temporal patterns specific to each borough, potentially providing more accurate predictions.
def create_train_test_data(journey_train_scaled, journey_test_scaled, lookback):
    # index is last element of journey_train_scaled
    demand_index = journey_train_scaled.columns.get_loc('demand')
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # No need to loop through each borough. We'll use all the data directly.
    borough_data_train = journey_train_scaled.values
    borough_data_test = journey_test_scaled.values

    for i in range(lookback, len(borough_data_train)):
        X_train.append(borough_data_train[i-lookback:i, :])
        Y_train.append(borough_data_train[i, demand_index])  

    for i in range(lookback, len(borough_data_test)):
        X_test.append(borough_data_test[i-lookback:i, :])
        Y_test.append(borough_data_test[i, demand_index])  

    # convert lists to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    return X_train, Y_train, X_test, Y_test



def create_lstm(X_train, units):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=units, return_sequences= True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
    lstm_model.add(LSTM(units=units, return_sequences= True))
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dense(units=1))
    return lstm_model


def transform_to_original_sclae(total_df, lstm_pred, Y_test, input_data, scaler):
    # transform to original scale
    demand_index = total_df.columns.get_loc('demand')
    dummy_array = np.zeros_like(input_data)

    dummy_array[-lstm_pred.shape[0]:, demand_index] = lstm_pred.ravel()
    predicted_lstm_inv = scaler.inverse_transform(dummy_array)[:, demand_index]

    dummy_array[-Y_test.shape[0]:, demand_index] = Y_test.ravel()
    Y_test_inv = scaler.inverse_transform(dummy_array)[:, demand_index]

    return predicted_lstm_inv, Y_test_inv




class Transformer:
    def __init__(self, num_heads=8, dropout_rate=0.1, num_layers=2):
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

    def create_transformer(self, input_shape):
        inputs = Input(shape=input_shape)

        x = inputs
        for _ in range(self.num_layers):
            # Self-attention and normalization
            attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=input_shape[-1])(x, x)
            attn_output = Dropout(self.dropout_rate)(attn_output)
            out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)


            # Feed-forward and normalization
            ffn_output = Dense(input_shape[-1], activation='relu')(out1)
            ffn_output = Dense(input_shape[-1])(ffn_output)
            ffn_output = Dropout(self.dropout_rate)(ffn_output)
            out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

            x = out2

        x = Flatten()(x)
        outputs = Dense(1)(x) # flatten and dense layer to produce scalar output -> encoder-only

        return Model(inputs=inputs, outputs=outputs)
    

    def train(self, x_train, y_train, batch_size=32, epochs=10, validation_split=0.1):
        model = self.create_transformer(x_train.shape[1:])
        model.compile(optimizer=Adam(), loss='mae') 

        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=4,
            restore_best_weights=True
        )

        history = model.fit(
            x_train, 
            y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split,  # Use 20% of data for validation
            callbacks=[early_stopping]  # Use EarlyStopping
        )

        self.model = model
        return history
    

    def predict(self, x):
        return self.model.predict(x)