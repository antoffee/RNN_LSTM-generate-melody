import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    """Создаем модель

    :param output_units (int): Количество единиц вывода
    :param num_units (list of int): Количество единиц в скрытых слоях
    :param loss (str): Тип функции потерь
    :param learning_rate (float): Скорость обучения для применения

    :return model (tf model): Модель
    """

    # создаем архитектуру модели
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # компилируем модель
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Тренируем и сохраняем модель

    :param output_units (int): Количество единиц вывода
    :param num_units (list of int): Количество единиц в скрытых слоях
    :param loss (str): Тип функции потерь
    :param learning_rate (float): Скорость обучения для применения
    """

    # генерация тренировочной последовательности
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # сборка сети
    model = build_model(output_units, num_units, loss, learning_rate)

    # тренируем модель
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # сохраняем модель
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()