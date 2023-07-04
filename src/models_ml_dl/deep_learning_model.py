from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.layers import MultiHeadAttention, Flatten
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

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

    boroughs = ['Westminster', 'Tower Hamlets', 'Kensington and Chelsea', 'Camden', 'Hammersmith and Fulham', 'Lambeth', 'Wandsworth', 'Southwark', 
                'Hackney', 'City of London', 'Islington', 'Newham']

    borough_indices = [journey_train_scaled.columns.get_loc('start_borough_' + borough) for borough in boroughs]

    # Convert dataframes to numpy arrays
    journey_train_scaled = journey_train_scaled.values
    journey_test_scaled = journey_test_scaled.values

    for i in range(len(journey_train_scaled)):
        X_train_temp = [] # create a temporary list to store the current sequence
        current_borough = np.argmax(journey_train_scaled[i, borough_indices]) # identify the borough of the current data point
        X_train_temp.append(journey_train_scaled[i]) # add the current data point to the sequence

        for j in range(i+1, len(journey_train_scaled)): # iterate over the rest of the data points starting from the next data point
            if np.argmax(journey_train_scaled[j, borough_indices]) == current_borough: # if the borough of the current data point (j) is the same as the borough of the initial data point (i)
                X_train_temp.append(journey_train_scaled[j]) # add the current data point (j) to the sequence
                if len(X_train_temp) == lookback + 1: # if the sequence has reached the desired length (lookback + 1)
                    X_train.append(np.array(X_train_temp[:-1])) # add the sequence (excluding the last data point) to the training input data
                    Y_train.append(journey_train_scaled[i+lookback, demand_index]) # ddd the demand of the last data point in the sequence to the training output data
                    break # break the inner loop as we have collected a complete sequence for training

    for i in range(len(journey_test_scaled)):
        X_test_temp = []
        current_borough = np.argmax(journey_test_scaled[i, borough_indices])
        X_test_temp.append(journey_test_scaled[i])

        for j in range(i+1, len(journey_test_scaled)):
            if np.argmax(journey_test_scaled[j, borough_indices]) == current_borough:
                X_test_temp.append(journey_test_scaled[j])
                if len(X_test_temp) == lookback + 1:
                    X_test.append(np.array(X_test_temp[:-1])) #exclude the last one
                    Y_test.append(journey_test_scaled[i+lookback, demand_index])
                    break

    # Convert lists to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Return arrays
    return X_train, Y_train, X_test, Y_test




def create_lstm(X_train, units, dropout, reg):
    lstm_model = Sequential()
    # first lstm layer
    lstm_model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(Dropout(dropout))
    # second lstm layer
    lstm_model.add(LSTM(units=units, return_sequences=True))
    lstm_model.add(Dropout(dropout))
    # third lstm layer
    lstm_model.add(LSTM(units=units, return_sequences=True))
    lstm_model.add(Dropout(dropout))
    # forth lstm layer
    #lstm_model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(reg)))
    #lstm_model.add(Dropout(dropout))
    # fifth lstm layer
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dropout(dropout))
    # output layer
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
            patience=10,
            restore_best_weights=True
        )

        # Create a ModelCheckpoint callback
        checkpoint = ModelCheckpoint(
            "./transformer_model_epoch_{epoch}.h5",  
            monitor="val_loss",  
            verbose=1,  
            save_best_only=False, 
            mode="auto",  
            save_freq="epoch",
        )

        history = model.fit(
            x_train, 
            y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split, 
            callbacks=[early_stopping, checkpoint]
        )

        self.model = model
        return history
    

    def predict(self, x):
        return self.model.predict(x)
