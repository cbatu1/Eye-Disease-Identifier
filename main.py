import cv2
from keras.models import load_model
import os
import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

CATEGORIES = ["Healthy_Eyes", "Bulging_Eyes", "Cataracts", "Crossed_Eyes", "Glaucoma", "Uveitis"]

model = load_model("64x0-CNN.model")


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MainPage(FloatLayout):
    loadfile = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        open(os.path.join(path, filename[0]))
        self.dismiss_popup()

    def predictDisease(self, filename):
        prediction = model.predict([prepare(filename)])
        return CATEGORIES[int(prediction[0][0])]


class EyeDiseaseIdentifierApp(App):
    pass


if __name__ == '__main__':
    EyeDiseaseIdentifierApp().run()
