import tensorflow as tf
from tensorflow.keras import layers, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predictor:
    def __init__(self, input_dim, hidden_units=[64, 32], num_classes=3):
        self.model = self.build_model(input_dim, hidden_units, num_classes)

    def build_model(self, input_dim, hidden_units=[64, 32], num_classes=3):
        inputs = tf.keras.Input(shape=(input_dim,), name="features")
        x = inputs
        for i, units in enumerate(hidden_units, start=1):
            x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        outputs = layers.Dense(num_classes, name="logits")(x)
        model = Model(inputs=inputs, outputs=outputs, name="fit_predictor")
        return model

    def compile(self, learning_rate=1e-3):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def fit(self, dataset, epochs=10, **fit_kwargs):
        return self.model.fit(dataset, epochs=epochs, **fit_kwargs)

    def predict(self, x):
        return self.model(x, training=False)

    def save_model(self, filepath):
        raise NotImplementedError
