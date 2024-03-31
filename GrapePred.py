from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd 
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

root = Tk()
root.title("GrapePred")
root.geometry('350x500')
root.resizable(False, False)
icon = PhotoImage(file="main.png")
root.iconphoto(False, icon)

kn = 0
img_n = "1.jpg"
size = 224


grape4 = "GrapeMax4_Model.hdf5"
grape7 = "Grape_Model.hdf5"


class_list = ['Черная гниль', 'Бактериальное увядание','Здоровый' ,'Черная пятнистость']
model_choice = grape4

def callback():
    global kn
    global label_img
    global img_n
    if(kn == 0):
        kn = 1
    else:
        kn = 0
        label_img.destroy()
    img_n = fd.askopenfilename() 
    label_img.config(text = img_n)

def pred_info():
    global model_choice
    global class_list
    global size
    global img_n
    if lang.get() == "GrapeMax4_Model.hdf5":
        model_choice = grape4
        class_list = ['Черная гниль', 'Бактериальное увядание','Здоровый' ,'Черная пятнистость']
        size = 224
    if lang.get() == "Grape_Model.hdf5":
        model_choice = grape7
        class_list = ['Черная гниль', 'Оидиум', 'Милдью', 'Бактериальное увядание', 'Краснуха','Здоровый' ,'Черная пятнистость']
        size = 160
    textVar.delete('1.0', END)
    textVarMax.delete('1.0', END)
    Grape_BaseVGG16 = keras.models.load_model(model_choice)
    list_of_classes = class_list

    img_path = img_n
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = Grape_BaseVGG16.predict(x)
    for i in range(len(list_of_classes)):
        stroka =list_of_classes[i] + ' : '
        textVar.insert(END,stroka)
        textVar.insert(END, str(preds[0][i])+'\n')
    textVarMax.insert(END, list_of_classes[np.argmax(preds[0])] + " : ")
    textVarMax.insert(END, preds[0][np.argmax(preds[0])])


lang = StringVar(value = grape4)

header = ttk.Label(text="Выберите модель для распознования: ")
header.pack(**{"padx":50, "pady":5, "anchor":NW})

grape7_btn = ttk.Radiobutton(text=grape7, value=grape7, variable= lang)
grape7_btn.pack(**{"padx":50, "pady":1, "anchor":CENTER})

grape4_btn = ttk.Radiobutton(text=grape4, value=grape4, variable= lang)
grape4_btn.pack(**{"padx":50, "pady":1, "anchor":CENTER})

button_dir = ttk.Button(text='Выберите картинку с листом винограда', command=callback)
button_dir.pack(**{"padx":50, "pady":5, "anchor":NW})

label_text = ttk.Label(text="Ссылка на изображение:")
label_text.pack(**{"padx":50, "pady":1, "anchor":NW})

label_img = ttk.Label(text = "ПУСТАЯ СТРОКА")
label_img.pack(**{"padx":50, "pady":5, "anchor":NW})

textVar = Text(width = 30, height = 14, wrap=WORD)
textVar.pack(**{"padx":50, "pady":1, "anchor":NW})

button_start = ttk.Button(text='Узнать болезнь', command=pred_info)
button_start.pack(**{"padx":50, "pady":5, "anchor":CENTER})

labelMax = ttk.Label(text = "Болезнь?")
labelMax.pack(**{"padx":50, "pady":1, "anchor":NW})

textVarMax = Text(width = 30, height = 2, wrap=WORD)
textVarMax.pack(**{"padx":50, "pady":1, "anchor":NW})

root.mainloop()