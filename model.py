import numpy as np
import pandas as pd
import tensorflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

class modelNet(keras.Model):
    def __init__(self, num_classes):
        super(modelNet, self).__init__()

        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')

        self.dropout = layers.Dropout(rate=0.5)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):

        x = self.dense1(inputs)
        x = self.dense2(x)
        
        x = self.dropout(x, training=training)
        
        x = self.dense3(x)

        output = self.output_layer(x)

        return output

model = modelNet(num_classes=10)
dummy_data = tensorflow.random.normal((2, 20))
out = model(dummy_data, training=False)
print(out.shape)