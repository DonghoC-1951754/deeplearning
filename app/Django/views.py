# filename: views.py (Django view functions)
from django.shortcuts import render
from django.http import HttpResponse

from django import forms


class ImgForm(forms.Form):
    img_field = forms.ImageField(widget=forms.FileInput)


def index(request):
    context = {}
    context['imgForm'] = ImgForm()

    if request.method == 'POST':
        form = ImgForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['img_field']
            # TODO: add ai here

            # results is a list of tool names and confidence levels
            context['results'] = []
            context['results'].append({'tool': img, 'confidence': 0.9})
            context['results'].append({'tool': 'a', 'confidence': 0.3})
            context['results'].append({'tool': 'img', 'confidence': 0.001})

    return render(request, "../templates/index.html", context)
