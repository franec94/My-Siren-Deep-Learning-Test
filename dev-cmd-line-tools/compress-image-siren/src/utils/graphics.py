from __future__ import print_function
from __future__ import division

# Standard Library, plus some Third Party Libraries
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple

import configargparse
from functools import partial

import copy
import datetime
import h5py
import math
import os
import random
import sys
import time
# import visdom

# Data Science and Machine Learning Libraries
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def show_graphic_series_via_plot(
    series,
    image_path = "graphic_series.png", figsize=(10, 7),
    title = "Series Graphic",
    color='orange', label='data series',
    xlabel = 'pos', ylabel = 'value',
    show_min_max = False,
    ):

    plt.figure(figsize=figsize)
    if show_min_max is True:
        max_value = series.max()
        max_pos = series.argmax()
        plt.annotate(f"epoch = {max_pos:.0f} | val = {max_value:.3f}", (max_pos, max_value))
        # plt.annotate(f"epoch = {max_pos:.0f}", (max_pos, 0))
        plt.vlines(x = max_pos,  ymin = 0, ymax = max_value,  linestyles='dashed')


        middle_value = series[len(series)//2]
        middle_pos = len(series)//2
        plt.annotate(f"epoch = {middle_pos:.0f} | val {middle_value:.3f}", (middle_pos, middle_value))
        plt.vlines(x = middle_pos,  ymin = 0, ymax = middle_value,  linestyles='dashed')

        min_value = series.min()
        min_pos = series.argmin()
        plt.annotate(f"epoch = {min_pos:.0f} | val {min_value:.3f}", (min_pos, min_value))
        # plt.annotate(f"epoch = {min_pos:.0f}", (min_pos, 0))
        plt.vlines(x = min_pos,  ymin = 0, ymax = min_value,  linestyles='dashed')
        pass
    plt.plot(series, color=color, label=label)
    # plt.plot(val_loss, color='red', label='validataion loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(image_path)
    plt.show()
    pass

def show_graphic_series_via_ax(
    series,
    ax,
    image_path = "graphic_series.png", figsize=(10, 7),
    title = "Series Graphic",
    color='orange', label='data series',
    xlabel = 'pos', ylabel = 'value',
    show_min_max = False,
    ):

    if show_min_max is True:
        max_value = series.max()
        max_pos = series.argmax()
        ax.annotate(f"epoch = {max_pos:.0f} | val {max_value:.3f}", (max_pos, max_value))

        ax.vlines(x = max_pos,  ymin = 0, ymax = max_value,  linestyles='dashed')

        middle_value = series[len(series)//2]
        middle_pos = len(series)//2
        ax.annotate(f"epoch = {middle_pos:.0f} | val {middle_value:.3f}", (middle_pos, middle_value))
        ax.vlines(x = middle_pos,  ymin = 0, ymax = middle_value,  linestyles='dashed')

        min_value = series.min()
        min_pos = series.argmin()
        ax.annotate(f"epoch = {min_pos:.0f} | val {min_value:.3f}", (min_pos, min_value))

        ax.vlines(x = min_pos,  ymin = 0, ymax = min_value,  linestyles='dashed')
        pass

    ax.plot(series, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    pass

def show_graphic_series(
    series,
    image_path = "graphic_series.png", figsize=(10, 7),
    title = "Series Graphic",
    color='orange', label='data series',
    xlabel = 'pos', ylabel = 'value',
    ax = None,
    show_min_max = False,
    ):
    if ax is None:
        show_graphic_series_via_plot(
        series = series,
        image_path = image_path,
        title = title,
        figsize = figsize,
        color = color,
        label = label,
        xlabel = xlabel,
        ylabel = ylabel,
        show_min_max = show_min_max
    )
    else:
        show_graphic_series_via_ax(
        series = series,
        ax = ax,
        image_path = image_path,
        title = title,
        figsize = figsize,
        color = color,
        label = label,
        xlabel = xlabel,
        ylabel = ylabel,
        show_min_max = show_min_max
    )
    pass

def plot_series_graphic_by_config(series, config):
    show_graphic_series(
        series = series,
        image_path = config.image_path,
        title = config.title,
        figsize = config.figsize,
        color = config.color,
        label = config.label,
        xlabel = config.xlabel,
        ylabel = config.ylabel,
        ax = config.ax,
        show_min_max = config.show_min_max
    )
    pass

def plot_loss_graphic(loss_data, config):
    show_graphic_series(
        series = loss_data,
        image_path = config.image_path,
        title = config.title,
        figsize = config.figsize,
        color = config.color,
        label = config.label,
        xlabel = config.xlabel,
        ylabel = config.ylabel,
        show_min_max = config.show_min_max
    )
    pass

def plot_psnr_graphic(psnr_data, config):
    show_graphic_series(
        series = psnr_data,
        image_path = config.image_path,
        title = config.title,
        figsize = config.figsize,
        color = config.color,
        label = config.label,
        xlabel = config.xlabel,
        ylabel = config.ylabel,
        show_min_max = config.show_min_max
    )
    pass

def show_image_with_metrcis_scores(image, sidelenght, metrics_txt):
    fig = plt.figure(figsize = (10, 9))

    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    # print(msg)

    ax = fig.add_subplot(111)
    ax.text(sidelenght//2, sidelenght//2, metrics_txt,
        # transform=ax.transAxes,
        fontsize=15,
        color='white',
        verticalalignment='bottom', horizontalalignment='right',)

    plt.imshow(image)
    plt.savefig('/content/predicted_images.png')
    plt.show()
    pass