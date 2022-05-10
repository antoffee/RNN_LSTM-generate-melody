import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# длительности
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]


def load_songs_in_kern(dataset_path):
    """Загружает все фрагменты kern в набор данных с помощью music21.

    :param dataset_path (str): Путь до датасета
    :return songs (list of m21 streams): Список(List) всех фрагментов
    """
    songs = []

    # просмотреть все файлы в наборе данных и загрузить их с помощью music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # содержит только kern файлы
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    """Логическая подпрограмма, которая возвращает True, если фрагмент имеет всю допустимую длительность, и False в противном случае.

    :param song (m21 stream): Мелодия
    :param acceptable_durations (list): Список допустимой продолжительности в четверти длины
    :return (bool):
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """Транспонирует песню в до мажор/ля минор (так как в этих тональностях отсутствуют бемоли и диезы)

    :param piece (m21 stream): Часть для транспонирования
    :return transposed_song (m21 stream): Транспонированная мелодия
    """

    # получаем ключ из песни
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # оцениваем ключ с помощью music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # находим интервал для транспонирования, например Си мажор (B) -> До мажор (С)
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # транспонируем мелодию с посчитанным интервалом
    tranposed_song = song.transpose(interval)
    return tranposed_song


def encode_song(song, time_step=0.25):
    """Преобразует партитуру в музыкальное представление, похожее на временной ряд. Каждый элемент в
    закодированном списке представляет наименьшую длительность
    четверти. Символы, используемые на каждом шаге: целые числа для MIDI-нот, «r» для обозначения паузы и «_»
    для представления нот/пауз, переносимых на новый временной шаг. Вот пример кодировки:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): часть для кодирования
    :param time_step (float): Продолжительность каждого временного шага в четверти длины
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # обрабатываем ноты
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # обрабатываем паузы
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # переводим ноту или паузу в нотацию временного ряда
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # Если мы впервые видим ноту/паузу, то закодируем ее. В противном случае, это означает, 
            # что мы повторяем то же самое
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # переводим песню в строку
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):

    # подгружаем песни
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # фильтруем песни с недопустимой продолжительностью
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # транспонируем в Cmaj/Amin
        song = transpose(song)

        # кодируем мелодии
        encoded_song = encode_song(song)

        # сохраняем в текстовый файл
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Создаем файл, в котором сопоставляются все закодированные песни и добавляются новые разделители фрагментов.

    :param dataset_path (str): путь до папки
    :param file_dataset_path (str): путь до файла
    :param sequence_length (int): кол-во временных шагов, которые необходимо учитывать при обучении
    :return songs (str): строка с всем датасетом и разделителями
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers

    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # загружаем маппинг
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # переводим строку в числа
    songs = songs.split()

    # переводим мелодию в числа
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    """Создаем образцы входных и выходных данных для обучения. Каждый образец представляет собой последовательность.

    :param sequence_length (int): Длина каждой последовательности. С квантованием на 16 нотах 64 ноты равняются 4 тактам.

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # подгружаем мелодии и переводим в числа
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # генерируем тренировочные последовательности
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot кодирование последовательностей
    vocabulary_size = len(set(int_songs))
    # размер входных данных: (количество последовательностей, длина последовательности, размер словаря)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()


