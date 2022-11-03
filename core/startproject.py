import os, sys, platform
import glob
import shutil
from datetime import datetime
import itertools
import datetime, time
import json

import math
import numpy as np
from collections.abc import Iterable

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import display, Image, Markdown, HTML
from dotenv import load_dotenv
import core.config as config

__version__ = config.VERSION

datasets_dir = None
notebook_id = None
running_mode = None
run_dir = None

_save_figs = True
_figs_dir = './figs'
_figs_name = 'fig_'
_figs_id = 0

_start_time = None
_end_time = None
_chrono_start = None
_chrono_stop = None
load_dotenv()

# -------------------------------------------------------------
# init_all
# -------------------------------------------------------------
#


def load_cssfile(cssfile):
    if cssfile is None: return
    styles = open(cssfile, "r").read()
    display(HTML(styles))


def init(name=None, run_directory='./run'):
    global notebook_id
    global datasets_dir
    global run_dir
    global _start_time

    # ---- Parameters from config.py
    #
    notebook_id = config.DEFAULT_NOTEBOOK_NAME if name is None else name
    mplstyle = config.PROJECT_MPLSTYLE
    cssfile = config.PROJECT_CSSFILE

    # ---- Load matplotlib style and css
    #
    matplotlib.style.use(mplstyle)
    load_cssfile(cssfile)
    datasets_dir = os.getenv('EMPLOYEES_TURNOVER_DATASET', False)
    if datasets_dir is False:
        error_datasets_not_found()
    # datasets_dir=os.path.expanduser("/data/AllProjectIA/EmployeesTurnover/dataset")

    # ---- run_dir
    #
    attrs = override('run_dir', return_attributes=True)
    run_dir = attrs.get('run_dir', run_directory)
    os.mkdir(run_dir)
    # ---- Tensorflow log level
    #
    log_level = int(os.getenv('TF_CPP_MIN_LOG_LEVEL', 0))
    str_level = ['Info + Warning + Error', 'Warning + Error', 'Error only'][log_level]

    # ---- Today, now and hostname
    #
    _start_time = datetime.datetime.now()
    h = platform.uname()

    # ---- Hello world
    #
    display_md('<br>**PROJECT 2022 - EMPLOYEE TURNOVER**')
    print('Version              :', config.VERSION)
    print('Notebook id          :', notebook_id)
    print('Run time             :', _start_time.strftime("%A %d %B %Y, %H:%M:%S"))
    print('Hostname             :', f'{h[1]} ({h[0]})')
    print('Tensorflow log level :', str_level, f' (={log_level})')
    print('Datasets dir         :', datasets_dir)
    print('Run dir              :', run_dir)

    # ---- Versions catalog
    #
    for m in config.USED_MODULES:
        if m in sys.modules:
            print(f'{m:21s}:', sys.modules[m].__version__)

    # ---- Save figs or not
    #
    save_figs = os.getenv('EMPLOYEES_TURNOVER_SAVE_FIGS', str(config.SAVE_FIGS))
    if save_figs.lower() == 'true':
        set_save_fig(save=True, figs_dir=f'{run_dir}/figs', figs_name='fig_', figs_id=0)

    return datasets_dir


# ------------------------------------------------------------------
# Where are my datasets ?
# ------------------------------------------------------------------
#
def error_datasets_not_found():
    display_md('## ATTENTION !!\n----')
    print('Le dossier contenant les datasets est introuvable\n')
    print('Pour que les notebooks puissent les localiser, vous devez :\n')
    print('         1/ Récupérer le dossier datasets')
    print('            Une archive (datasets.tar) est disponible via le repository kaggle.\n')
    print("         2/ Préciser la localisation de ce dossier datasets via la variable")
    print("            d'environnement : EMPLOYEES_TURNOVER_DATASET_DIR.\n")
    print('Exemple :')
    print("   Dans votre fichier .env :")
    print("   EMPLOYEES_TURNOVER_DATASET_DIR='/data/AllProjectIA/EmployeesTurnover/dataset")
    display_md('----')
    assert False, 'datasets folder not found, please set EMPLOYEES_TURNOVER_DATASET env var.'


def override(*names, module_name='__main__', verbose=True, return_attributes=False):
    module = sys.modules[module_name]
    if len(names) == 0:
        names = []
        for name in dir(module):
            if name.startswith('_'): continue
            v = getattr(module, name)
            if type(v) not in [str, int, float, bool, tuple, list, dict]: continue
            names.append(name)

    # ---- Search for names
    #
    overrides = {}
    for name in names:

        # ---- Environment variable name
        #
        env_name = f'PROJECT_OVERRIDE_{notebook_id}_{name}'
        env_value = os.environ.get(env_name)

        # ---- Environment variable : Doesn't exist
        #
        if env_value is None: continue

        # ---- Environment variable : Exist
        #
        value_old = getattr(module, name)
        value_type = type(value_old)

        if value_type in [str]:
            new_value = env_value.format(datasets_dir=datasets_dir, notebook_id=notebook_id)

        if value_type in [int, float, bool, tuple, list, dict, type(None)]:
            new_value = eval(env_value)

        # ---- Override value
        #
        setattr(module, name, new_value)
        overrides[name] = new_value

    if verbose and len(overrides) > 0:
        display_md('**\*\* Overrided parameters : \*\***')
        for name, value in overrides.items():
            print(f'{name:20s} : {value}')

    if return_attributes:
        return overrides


def np_print(*args, precision=3, linewidth=120):
    with np.printoptions(precision=precision, linewidth=linewidth):
        for a in args:
            print(a)


def display_md(text):
    display(Markdown(text))


def subtitle(t):
    display(Markdown(f'<br>**{t}**'))


def display_html(text):
    display(HTML(text))


def display_img(img):
    display(Image(img))


def set_save_fig(save=True, figs_dir='./run/figs', figs_name='fig_', figs_id=0):
    """
    Set save_fig parameters
    Default figs name is <figs_name><figs_id>.{png|svg}
    args:
        save      : Boolean, True to save figs (True)
        figs_dir  : Path to save figs (./figs)
        figs_name : Default basename for figs (figs_)
        figs_id   : Start id for figs name (0)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    _save_figs = save
    _figs_dir = figs_dir
    _figs_name = figs_name
    _figs_id = figs_id
    print(f'Save figs            : {_save_figs}')
    print(f'Path figs            : {_figs_dir}')


def update_progress(what, i, imax, redraw=False, verbosity=1):
    if verbosity == 0:   return
    if verbosity == 2 and i < imax: return
    bar_length = min(40, imax)
    if (i % int(imax / bar_length)) != 0 and i < imax and not redraw:
        return
    progress = float(i / imax)
    block = int(round(bar_length * progress))
    endofline = '\r' if progress < 1 else '\n'
    text = "{:16s} [{}] {:>5.1f}% of {}".format(what, "#" * block + "-" * (bar_length - block), progress * 100, imax)
    print(text, end=endofline)


def chrono_start():
    global _chrono_start, _chrono_stop
    _chrono_start = time.time()


# return delay in seconds or in humain format
def chrono_stop(hdelay=False):
    global _chrono_start, _chrono_stop
    _chrono_stop = time.time()
    sec = _chrono_stop - _chrono_start
    if hdelay: return hdelay_ms(sec)
    return sec


def chrono_show():
    print('\nDuration : ', hdelay_ms(time.time() - _chrono_start))


def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))


def mkdir(path):
    os.makedirs(path, mode=0o750, exist_ok=True)


def tag_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


# Return human delay like 01:14:28 543ms
# delay can be timedelta or seconds
def hdelay_ms(delay):
    if type(delay) is not datetime.timedelta:
        delay = datetime.timedelta(seconds=delay)
    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh * 3600 - mm * 60
    ms = (sec - int(sec)) * 1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'


def hsize(num, suffix='o'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Y{suffix}'


def shuffle_np_dataset(x, y):
    """
    Shuffle a dataset (x,y)
    args:
        x,y : dataset
    return:
        x,y mixed
    """
    assert (len(x) == len(y)), "x and y must have same size"
    np.random.seed(42)
    p = np.random.permutation(len(x))
    return x[p], y[p]


def rescale_dataset(*data, scale=1):
    return [d[:int(scale * len(d))] for d in data]


def pick_dataset(*data, n=5):
    ii = np.random.choice(range(len(data[0])), n)
    out = [d[ii] for d in data]
    return out[0] if len(out) == 1 else out


def set_save_fig(save=True, figs_dir='./run/figs', figs_name='fig_', figs_id=0):
    """
    Set save_fig parameters
    Default figs name is <figs_name><figs_id>.{png|svg}
    args:
        save      : Boolean, True to save figs (True)
        figs_dir  : Path to save figs (./figs)
        figs_name : Default basename for figs (figs_)
        figs_id   : Start id for figs name (0)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    _save_figs = save
    _figs_dir = figs_dir
    _figs_name = figs_name
    _figs_id = figs_id
    print(f'Save figs            : {_save_figs}')
    print(f'Path figs            : {_figs_dir}')


def save_fig(filename='auto', png=True, svg=False):
    """
    Save current figure
    args:
        filename : Image filename ('auto')
        png      : Boolean. Save as png if True (True)
        svg      : Boolean. Save as svg if True (False)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    if filename is None: return
    if not _save_figs: return
    if not os.path.exists(_figs_dir):
        os.mkdir(_figs_dir)
    if filename == 'auto':
        path = f'{_figs_dir}/{notebook_id}-{_figs_name}{_figs_id:02d}'
    else:
        path = f'{_figs_dir}/{notebook_id}-{filename}'
    if png: plt.savefig(f'{path}.png')
    if svg: plt.savefig(f'{path}.png')
    if filename == 'auto': _figs_id += 1
    display_html(f'<div class="comment">Saved: {path}</div>')


def plot_images(x, y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                fontsize=20, interpolation='lanczos', save_as='auto'):
    if indices == 'all': indices = range(len(x))
    if norm and len(norm) == 2: norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    draw_labels = (y is not None)
    draw_pred = (y_pred is not None)
    rows = math.ceil(len(indices) / columns)
    fig = plt.figure(figsize=(columns * x_size, rows * (y_size + y_padding)))
    n = 1
    for i in indices:
        axs = fig.add_subplot(rows, columns, n)
        n += 1
        # ---- Shape is (lx,ly)
        if len(x[i].shape) == 2:
            xx = x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape) == 3:
            (lx, ly, lz) = x[i].shape
            if lz == 1:
                xx = x[i].reshape(lx, ly)
            else:
                xx = x[i]
        img = axs.imshow(xx, cmap=cm, norm=norm, interpolation=interpolation)
        #         img=axs.imshow(xx,   cmap = cm, interpolation=interpolation)
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i], fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i] != y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})', fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i], fontsize=fontsize)
        if colorbar:
            fig.colorbar(img, orientation="vertical", shrink=0.65)
    save_fig(save_as)
    plt.show()


def plot_image(x, cm='binary', figsize=(4, 4), interpolation='lanczos', save_as='auto'):
    if len(x.shape) == 2:
        xx = x
    # ---- Shape is (lx,ly,n)
    if len(x.shape) == 3:
        (lx, ly, lz) = x.shape
        if lz == 1:
            xx = x.reshape(lx, ly)
        else:
            xx = x
    # ---- Draw it
    plt.figure(figsize=figsize)
    plt.imshow(xx, cmap=cm, interpolation=interpolation)
    save_fig(save_as)
    plt.show()


def plot_history(history, figsize=(8, 6),
                 plot={"Accuracy": ['accuracy', 'val_accuracy'], 'Loss': ['loss', 'val_loss']},
                 save_as='auto'):
    fig_id = 0
    for title, curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        if save_as == 'auto':
            figname = 'auto'
        else:
            figname = f'{save_as}_{fig_id}'
            fig_id += 1
        save_fig(figname)
        plt.show()


def plot_confusion_matrix(y_true, y_pred,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figsize=(10, 8),
                          digit_format='{:0.2f}',
                          save_as='auto'):
    cm = confusion_matrix(y_true, y_pred, normalize=None, labels=target_names)

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, digit_format.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    save_fig(save_as)
    plt.show()
