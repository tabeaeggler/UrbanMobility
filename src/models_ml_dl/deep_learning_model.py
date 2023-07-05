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
from sklearn.preprocessing import MinMaxScaler



######## DATA PREPARATION ########

def create_train_test_data(journey_train_scaled, journey_test_scaled, lookback, boroughs):
    """
    This function creates the input and output data for a DL model, that is designed to forecast bike sharing demand for each borough in London separately.
    It uses the concept of 'lookback', which is the number of past observations that the model uses to forecast the next one.
    
    Args:
    journey_train_scaled : DataFrame containing the scaled training data.
    journey_test_scaled : DataFrame containing the scaled testing data.
    lookback : int, Number of past observations to use for forecasting the next one.

    Returns:
    X_train : Array containing the input sequences for the training data.
    Y_train : Array containing the output sequences for the training data.
    X_test : Array containing the input sequences for the testing data.
    Y_test : Array containing the output sequences for the testing data.
    """

    # identify indices of boroughs and demand in the dataframe
    borough_indices = [journey_train_scaled.columns.get_loc('start_borough_' + borough) for borough in boroughs]
    demand_index = journey_train_scaled.columns.get_loc('demand')

    # convert dataframes to numpy arrays
    journey_train_scaled = journey_train_scaled.values
    journey_test_scaled = journey_test_scaled.values

    # initialize lists to store input sequences and corresponding outputs
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    # create training and testing sequences and corresponding outputs
    X_train, Y_train = _create_sequences(journey_train_scaled, borough_indices, demand_index, lookback, X_train, Y_train)
    X_test, Y_test = _create_sequences(journey_test_scaled, borough_indices, demand_index, lookback, X_test, Y_test)


    # convert lists to numpy arrays and return
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def _create_sequences(journey_scaled, borough_indices, demand_index, lookback, X, Y):
    """
    Helper function to create sequences and corresponding outputs.
    """
    for i in range(len(journey_scaled)):
        X_temp = [] # create a temporary list to store the current sequence
        current_borough = np.argmax(journey_scaled[i, borough_indices]) # identify the borough of the current data point
        X_temp.append(journey_scaled[i]) # add the current data point to the sequence

        for j in range(i+1, len(journey_scaled)): # iterate over the rest of the data points starting from the next data point
            if np.argmax(journey_scaled[j, borough_indices]) == current_borough: # if the borough of the current data point (j) is the same as the borough of the initial data point (i)
                X_temp.append(journey_scaled[j]) # add the current data point (j) to the sequence
                if len(X_temp) == lookback + 1: # if the sequence has reached the desired length (lookback + 1)
                    X.append(np.array(X_temp[:-1])) # add the sequence (excluding the last data point) to the training input data
                    Y.append(journey_scaled[i+lookback, demand_index]) # ddd the demand of the last data point in the sequence to the training output data
                    break # break the inner loop as we have collected a complete sequence for training
    return X, Y


def min_max_scaling(journey_train, journey_test, journey_train_orig, journey_test_orig):
    """
    This function scales the features in the journey data to a specified range (0 to 1) using Min-Max scaling.
    The scaler is fit on the training data and then used to transform both the training and test data. The 'demand' feature is then added back to the scaled data.

    Args:
    journey_train : DataFrame containing the journey training data.
    journey_test : DataFrame containing the journey testing data.
    journey_train_orig : Original (unscaled) DataFrame containing the journey training data.
    journey_test_orig : Original (unscaled) DataFrame containing the journey testing data.

    Returns:
    journey_train_scaled : DataFrame containing the scaled journey training data with 'demand' added back.
    journey_test_scaled : DataFrame containing the scaled journey testing data with 'demand' added back.
    """

    # initialize the Min-Max scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit the scaler using the training data and transform the training data
    journey_train_scaled = scaler.fit_transform(journey_train)

    # use the fitted scaler to transform the test data
    journey_test_scaled = scaler.transform(journey_test)

    # convert the scaled arrays back to DataFrames
    journey_train_scaled = pd.DataFrame(journey_train_scaled, columns=journey_train.columns)
    journey_test_scaled = pd.DataFrame(journey_test_scaled, columns=journey_test.columns)

    # add the 'demand' feature back to the scaled data
    journey_train_scaled['demand'] = journey_train_orig['demand'].values
    journey_test_scaled['demand'] = journey_test_orig['demand'].values

    return journey_train_scaled, journey_test_scaled



def positional_encoding(length, d_model):
    """
    Compute positional encoding for a given sequence length and model dimension.
    
    Args:
    length (int): Length of the sequence.
    d_model (int): Dimension of the model.

    Returns:
    numpy.ndarray: Positional encoding matrix.
    """
    pos_enc = np.zeros((length, d_model))
    for pos in range(length):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))

    return pos_enc


def add_positional_encoding(data):
    """
    Add positional encoding to the input data.

    Args:
    data (numpy.ndarray): The input data with 3D tensor shape (num_samples, sequence_length, num_features).

    Returns:
    numpy.ndarray: The input data with added positional encoding.
    """
    num_samples, sequence_length, num_features = data.shape
    pos_enc = positional_encoding(sequence_length, num_features)

    # expand dims to match shape of data
    pos_enc = np.expand_dims(pos_enc, axis=0)

    # repeat positional encoding for each sample
    pos_enc = np.repeat(pos_enc, num_samples, axis=0)

    return data + pos_enc




######## LSTM MODEL ########

def create_lstm(X_train, units, dropout, reg):
    """
    Create LSTM model with four LSTM layers and a dense output layer.
    
    Args:
    X_train (numpy.ndarray): The training data with shape (num_samples, sequence_length, num_features).
    units (int): The number of LSTM units for each layer.
    dropout (float): The dropout rate for the dropout layer after each LSTM layer.
    reg (float): The regularization factor. (Not used in this function, but can be used to add regularization to the LSTM layers)

    Returns:
    keras.models.Sequential: The constructed LSTM model.
    """

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
    lstm_model.add(LSTM(units=units))
    lstm_model.add(Dropout(dropout))
    # output layer
    lstm_model.add(Dense(units=1))
    return lstm_model




######## TRANSFORMER MODEL ########

class Transformer:
    def __init__(self, num_heads=8, dropout_rate=0.1, num_layers=2):
        """
        Initialize Transformer model with given parameters.

        Args:
        num_heads (int): The number of attention heads. Default is 8.
        dropout_rate (float): The dropout rate for Dropout layers. Default is 0.1.
        num_layers (int): The number of Transformer layers. Default is 2.
        """
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

    def create_transformer(self, input_shape):
        """
        Create a Transformer model with the previously specified parameters.

        Args:
        input_shape (tuple): The shape of the input data.

        Returns:
        tensorflow.python.keras.engine.training.Model: The constructed Transformer model.
        """
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
        outputs = Dense(1)(x) 

        return Model(inputs=inputs, outputs=outputs)

    def train(self, x_train, y_train, batch_size=32, epochs=10, validation_split=0.1):
        """
        Train the Transformer model on the given training data.

        Args:
        x_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training output data.
        batch_size (int): The batch size for training. Default is 32.
        epochs (int): The number of epochs to train for. Default is 10.
        validation_split (float): The fraction of the training data to be used as validation data. Default is 0.1.

        Returns:
        tensorflow.python.keras.callbacks.History: The history object that contains all information collected during training.
        """

        # create transfomer
        model = self.create_transformer(x_train.shape[1:])

        # add adam optimizer
        model.compile(optimizer=Adam(), loss='mae') 

        # add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True
        )

        # create a model checkpoint 
        checkpoint = ModelCheckpoint(
            "./transformer_model_epoch_{epoch}.h5",  
            monitor="val_loss",  
            verbose=1,  
            save_best_only=False, 
            mode="auto",  
            save_freq="epoch",
        )

        # fit model
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
        """
        Use the trained Transformer model to make predictions on the given data.

        Args:
        x (numpy.ndarray): The input data to make predictions on.

        Returns:
        numpy.ndarray: The predictions made by the model.
        """
        return self.model.predict(x)
