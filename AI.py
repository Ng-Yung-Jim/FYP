import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

num_nodes = 1000
X = np.array([]).reshape(10,num_nodes)
y = np.array([]).reshape(7,num_nodes)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.36, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(20/36), random_state=42)

def build_model(num_nodes, num_features):
    input_layer = tf.keras.layers.Input(shape=(num_nodes, 10))
    
    x = tf.keras.layers.Conv3D(128, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv3D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv3D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv3D(16, kernel_size=3, activation='relu', padding='same')(x)
    
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    
    output_layer = tf.keras.layers.Dense(num_nodes, activation='linear')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='SGD',
                  loss='mse',
                  metrics=['mae'])
    return model

model = build_model(num_nodes, 10)
model.summary()

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")