import os
from tkinter import *
from tkinter.ttk import Style

from melodygenerator import MelodyGenerator

from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

def play_melody():
  bashCommand = "timidity mel.mid"
  os.system(bashCommand)

def generate_melody():
    input_seed=txt.get()
    mg = MelodyGenerator()
    # from 50 to 80
    seed = "81 82 83 84 85"
    seed2 = "65 _ 64 _ 62 _ 60"
    melody = mg.generate_melody(input_seed, 500, SEQUENCE_LENGTH, 0.5)
    print(melody)
    mg.save_melody(melody)
    success_lbl= Label(window, text="Мелодия создана")
    success_lbl.grid(column=0, row=2)
    play_btn = Button(window, text="Воспроизвести мелодию", command=play_melody)
    play_btn.grid(column=0, row=3)

window = Tk()
window.title("Интерактивная программа выбора и синтеза музыки по нотам из заданного набора нот мелодий.")
window.geometry('800x500')

 
style = Style()
style.configure('TButton', font =
               ('calibri', 10, 'bold', 'underline'),
                foreground = 'red')

lbl = Label(window, text="Введите в инпут числа от 50 до 80 или \nсимволы _ через пробел для того, чтобы задать начало мелодии", font=("Arial Bold", 12))  
lbl.grid(column=0, row=0)  

txt = Entry(window,width=90)  
txt.grid(column=0, row=1)  
btn = Button(window, text="Сгенерировать мелодию", command=generate_melody)  
btn.grid(column=1, row=1) 

window.mainloop()