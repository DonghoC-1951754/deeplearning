# filename: views.py (Django view functions)
import os

import numpy as np
from django.shortcuts import render
from django.http import HttpResponse

from django import forms
import tensorflow as tf
import cv2
from PIL import Image


class ImgForm(forms.Form):
    img_field = forms.ImageField(widget=forms.FileInput)


model = tf.keras.models.load_model('./models/aug_9-1_4255acc.keras')


def index(request):
    context = {'imgForm': ImgForm()}

    if request.method == 'POST':
        form = ImgForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['img_field']
            img = np.frombuffer(img.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            results = model(img[None, :, :, :3])

            # results is a list of tool names and confidence levels
            context['results'] = []
            for idx, i in enumerate(['CombWrench', 'Hammer', 'Screwdriver', 'Wrench']):
                context['results'].append({'tool': i, 'confidence': tf.nn.softmax(results).numpy()[0][idx]})

            context['results'] = sorted(context['results'], key=lambda x: x['confidence'], reverse=True)

    return render(request, "../templates/index.html", context)


def info(request):
    return render(request, "../templates/info.html", {})
