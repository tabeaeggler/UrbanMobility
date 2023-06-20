from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.layers import MultiHeadAttention, Flatten
from keras.optimizers import Adam

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
        outputs = Dense(1)(x) 

        return Model(inputs=inputs, outputs=outputs)

    def train(self, x_train, y_train, batch_size=32, epochs=3):
        model = self.create_transformer(x_train.shape[1:])
        model.compile(optimizer=Adam(), loss='mean_squared_error') 

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        self.model = model

    def predict(self, x):
        return self.model.predict(x)
