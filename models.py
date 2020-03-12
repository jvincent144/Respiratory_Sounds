import tensorflow as tf

####
# Build a model class

class LSTM(tf.keras.Model):
    
    def __init__(self):
        super(LSTM, self).__init__()
        self.LSTM = tf.keras.layers.LSTM(8, input_shape = (44100, 1)) # Argument 1 : number of output features
        # Argument 2 : input_shape = # time points
        # input_shape is necessary for automatic shape inference
        self.dense = tf.keras.layers.Dense(8) # Argument 1 : number of output classes
        
    def call(self, data):
        data = self.LSTM(data)
        data = self.dense(data)
        data = tf.nn.softmax(data, axis = 1)
        return data