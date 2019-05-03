from modules.const import *
from modules.utility import show
from modules import card_detection
from modules import interpret_labels
from modules import nn_network


import cv2
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics import Color
from kivy.graphics import Rectangle

import os
  
def to_texture(image):
    image = cv2.flip(image, 0)
    im_bytes = np.reshape(image, [-1])
    out_texture = Texture.create(size=(image.shape[1], image.shape[0]))
    out_texture.blit_buffer(im_bytes, colorfmt='bgr', bufferfmt='ubyte')            

    return out_texture

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class ProbabilitiesLabel(Label):
    def __init__(self, prediction, color=(1.0, 1.0, 1.0, 1.0)):
        description = interpret_labels.get_description(prediction[0])
        probability = "{:2.0f}".format(prediction[1] * 100)      
        text = "{} ({})".format(description, probability)
        super().__init__(text=text, color=color)

class VotesLabel(Label):
    def __init__(self, prediction, color=(1.0, 1.0, 1.0, 1.0)):
        description = interpret_labels.get_description(prediction['classes'])
        votes = prediction['votes']
        #best_probability = "{:4.2f}".format(prediction['best_probability'] * 100)
        text = "{} ({}/4)".format(description, votes)
        super().__init__(text=text, color=color)

class CornerPredictions(BoxLayout):
    image = ObjectProperty(None)
    prediction_labels = ObjectProperty(None)

    def __init__(self, image, predictions=None, **kwargs):
        super().__init__(**kwargs)
        self.image.texture = to_texture(image)

    def set_predictions(self, predictions):
        self.prediction_labels.clear_widgets()
        predictions = [(i, p) for i, p in enumerate(predictions['probabilities'])]        
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        self.prediction_labels.add_widget(ProbabilitiesLabel(predictions[0], (0.0, 1.0, 0.0, 1.0)))
        for prediction in predictions[1:3]:
            self.prediction_labels.add_widget(ProbabilitiesLabel(prediction))

class CardPredictions(BoxLayout):
    ''' Predictions with probabilities for a single image'''
    bgcolor = NumericProperty(0.0)
    voting_labels = ObjectProperty(None)
    corner_predictions = ObjectProperty(None)
    def __init__(self, images, predictions=None,
                votes=None, color=0.0, **kwargs):
        super().__init__(**kwargs)                    
        self.bgcolor = color
        self.corner_widgets = []
        for image in images:
            new_corner = CornerPredictions(image)
            self.corner_widgets.append(new_corner)
            self.corner_predictions.add_widget(new_corner)
        if not predictions is None:
            self.set_predictions(predictions)
        
    def set_results(self, results):
        self.voting_labels.clear_widgets()
        self.voting_labels.add_widget(VotesLabel(results)) 
        for corner_widget, predictions in zip(self.corner_widgets, results['predictions']):
            corner_widget.set_predictions(predictions)  


class PredictionList(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.widget_list = []

    def add(self, new_widget):
        self.add_widget(new_widget)
        self.widget_list.append(new_widget)   

    def clear(self):
        self.clear_widgets()
        self.widget_list.clear()

    def get_widgets(self):
        return self.widget_list


class Screen(FloatLayout):
    image = ObjectProperty(None)
    text_output = ObjectProperty(None)
    corner_images = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        
    def dismiss_popup(self):
        self._popup.dismiss(animation=False)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.8), pos_hint={'y': 0.2},
                            auto_dismiss=False)
        self._popup.open()

    def load(self, path, filename):
        self.dismiss_popup()

        if not filename:
            return
        try:
            extracted_features = card_detection.for_classification(os.path.join(path, filename[0]))
            image = extracted_features['image']
            self.loaded_corners = extracted_features['corners']

            prepared = []
            for group in self.loaded_corners:
                prepared_group = []
                for im in group:
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    im = cv2.resize(im, (ISIZE, ISIZE))
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                    prepared_group.append(im)
                prepared.append(prepared_group)

            self.image.texture = to_texture(image)            
            self.corner_images.clear()
            for i, card_corners in enumerate(prepared):
                color = None
                if i % 2 == 0:
                    color = 0.2
                else:
                    color = 0.3
                new_widget = CardPredictions(card_corners, color=color)
                self.corner_images.add(new_widget)

            self.text_output.text = "Image loaded"

        except ValueError as e:
            self.text_output.text = str(e)

    def detect_hand(self):
        if self.loaded_corners is None:
            self.text_output.text = "Image couldn't be processed"
            return        

        predictions = nn_network.predict(self.loaded_corners)
        labels = [p['classes'] for p in predictions]
                
        for widget, results in zip(self.corner_images.get_widgets(), predictions):            
            widget.set_results(results)
        
        if len(self.loaded_corners) == 5:
            self.text_output.text = interpret_labels.get_combination(labels)
        else:
            self.text_output.text = "Couldn't find 5 cards in the image"


class GuiApp(App):
    def build(self):
        return Screen()
        
if __name__ == '__main__':
    GuiApp().run()
