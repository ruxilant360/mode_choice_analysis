from tensorflow import keras
from tensorflow.keras import layers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

def custom_Net(num_classes:int, learning_rate: float) -> tf.keras.Model:
    """
    builds model to predict classes, compiled with special metrics from tensorflow 
    """
    class modelNet(keras.Model):
        def __init__(self, num_classes):
            super(modelNet, self).__init__()
    
            self.dense1 = layers.Dense(256, activation='relu')
            self.dropout1 = layers.Dropout(rate=0.5)
            self.btcnorm1 = layers.BatchNormalization()
            
            self.dense2 = layers.Dense(128, activation='relu')
            self.dropout2 = layers.Dropout(rate=0.5)
            self.btcnorm2 = layers.BatchNormalization()
  
            self.dense3 = layers.Dense(64, activation='relu')
    
            self.output_layer = layers.Dense(num_classes, activation='softmax')
    
        def call(self, inputs, training=False):
            
            x = self.dense1(inputs)
            x = self.btcnorm1(x, training=training)
            x = self.dropout1(x, training=training)
            
            x = self.dense2(x)
            x = self.btcnorm2(x, training=training)
            x = self.dropout2(x, training=training)
            
            x = self.dense3(x)
            output = self.output_layer(x)
            return output

    model = modelNet(num_classes)
    model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                       loss="sparse_categorical_crossentropy",
                       metrics = ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")])

    return model