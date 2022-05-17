import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import os

class MelodyGenerator:
    """Класс, обертывающий модель LSTM и предлагающий утилиты для создания мелодий."""

    def __init__(self, model_path=os.path.abspath('model.h5')):
        """Конструктор, который инициализирует модель TensorFlow"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Генерирует мелодию с использованием модели и возвращает миди-файл.

        :param seed (str): Последовательность мелодии с обозначением, используемым для кодирования набора данных.
        :param num_steps (int): Количество шагов, которые необходимо сгенерировать
        :param max_sequence_len (int): Максимальное количество шагов в последовательности, которое будет учитываться при генерации
        :param temperature (float): Плавающая в интервале [0, 1]. Числа ближе к 0 делают модель более детерминированной.
            Число, близкое к 1, делает генерацию более непредсказуемой.

        :return melody (list of str): Список с символами, представляющими мелодию
        """

        # создаем последовательность с начальными символами
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # переводим в инт
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # обрезаем по максимальной длине последовательности
            seed = seed[-max_sequence_length:]

            # one-hot кодирование
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # сделать прогноз
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # обновляем последовательность
            seed.append(output_int)

            # сопоставить с нашей кодировкой
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # проверить, находимся ли мы в конце мелодии
            if output_symbol == "/":
                break

            # обновить мелодию
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Выбираем индекс из массива вероятностей, повторно применяя softmax с использованием температуры.

        :param predictions (nd.array): Массив, содержащий вероятности для каждого из возможных выходов.
        :param temperature (float): Плавающая в интервале [0, 1]. Числа ближе к 0 делают модель более детерминированной.
           Число, близкое к 1, делает генерацию более непредсказуемой.

        :return index (int): Выбранный выходной символ
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Переводим мелодию в MIDI файл

        :param melody (list of str): Мелодия
        :param min_duration (float): Продолжительность каждого временного шага в четверти длины
        :param file_name (str): Имя файла
        :return:
        """

        # создаем music21 поток
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # разбираем все символы в мелодии и создаем объекты ноты/паузы
        for i, symbol in enumerate(melody):

            # обрабатываем случай, в котором у нас есть нота/пауза
            if symbol != "_" or i + 1 == len(melody):

                # проверяем, что мы имеем дело с нотой/паузой после предыдущей
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # пауза
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # нота
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # сбрасываем счетчик шагов
                    step_counter = 1

                start_symbol = symbol

            # обрабатывать случай, когда у нас есть знак продления "_"
            else:
                step_counter += 1

        # записать поток m21 в миди-файл
        stream.write(format, file_name)


if __name__ == "__main__":
    arr = os.listdir('.')
    print(arr)
    mg = MelodyGenerator()
    # from 50 to 80
    seed = "67 68 69 70 71 72"
    seed2 = "65 _ 64 _ 62 _ 60"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.1)
    print(melody)
    mg.save_melody(melody)




















