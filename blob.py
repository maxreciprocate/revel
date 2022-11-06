import math
import random
import os, sys
import operator
from operator import mul, add, truediv
from math import pi, sqrt, tau
π = pi

import numpy as np
from numpy.random import rand, randint, randn, normal
from numpy import zeros, ones, empty, array, linspace, arange, prod
np.set_printoptions(formatter={'all': lambda x: str(x)})

from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from typing import Optional, Union, List, NamedTuple, Tuple

import itertools
from itertools import chain, product, repeat, count, combinations, zip_longest, starmap
from functools import partial, reduce, lru_cache
P = partial

import multiprocessing as mp
import subprocess
import shutil

def popen(cmd, *args, **kwargs):
    return subprocess.Popen(cmd, *args, shell=True, **kwargs)

from tqdm import tqdm, trange
from copy import deepcopy, copy
from datetime import datetime
from time import time, sleep
import pickle
import json
import toml
import csv
import heapq

def pprint(d: dict):
    for k,v in d.items():
        print(f'{k}: {v}')

ap = lambda f, xs: list(map(f, xs))

def log(xs):
    return np.log2(xs + 1e-24)

def normed(xs):
    return (xs - xs.mean()) / (xs.std() + 1e-300)

def flatten(xs):
    return list(reduce(lambda acc, x: acc + x, xs, []))

def findfirst(f, xs):
    for ind, vx in enumerate(xs):
        if f(vx):
            return ind

    return None

def findall(xs):
    return np.where(xs)[0]

def bench(fn, *args):
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp(fn)(*args)
    lp.print_stats(output_unit=1e-3)

class Heapin(NamedTuple):
    cost: float
    data: any

    def __lt__(self, o):
        return self.cost < o.cost

class Heap:
    def __init__(self, size):
        self.heap = []
        self.size = size

    def push(self, cost, data):
        h = Heapin(cost, data)

        if len(self.heap) < self.size:
            heapq.heappush(self.heap, h)
        else:
            heapq.heappushpop(self.heap, h)

    def __iter__(self):
        return iter(self.heap)

def nsplit(n: int, xs):
    out = [[] for _ in range(n)]

    for ind, x in enumerate(xs):
        out[ind % n].append(x)

    return out

def multicore(fn, args):
    if ncores == 1:
        return [fn(*list(args)[0])]

    try:
        pool = mp.Pool(ncores)

        if isinstance(args, list):
            out = pool.map(fn, args)
        else:
            out = pool.starmap(fn, args)
    finally:
        pool.close()
        pool.join()

    return out

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

ncores = 1 if run_from_ipython() else os.cpu_count() // 2

from matplotlib import pyplot
import matplotlib

matplotlib.rcParams["font.family"] = "Futura"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["xtick.labelsize"] = 18
matplotlib.rcParams["ytick.labelsize"] = 18
matplotlib.rcParams["figure.titlesize"] = 12
matplotlib.rcParams["figure.figsize"] = 15, 8

if run_from_ipython():
    matplotlib.rcParams["figure.dpi"] = 20
else:
    matplotlib.rcParams["figure.dpi"] = 100

matplotlib.style.use('ggplot')

def plot(*args, **kwargs):
    pyplot.plot(*args, **kwargs)

def barh(ts, label=None):
    return bar(ts, label, h=True)

def bar(ts, label=None, h=False):
    if label is None:
        label = np.arange(len(ts))

    if h:
        pyplot.barh(label[:len(ts)], ts)
    else:
        pyplot.bar(label[:len(ts)], ts)

def imshow(im, save=False, interpolation='bicubic', cmap='hot'):
    fig = pyplot.figure(figsize=(20, 100))
    ax = fig.gca()
    ax.imshow(im, cmap=cmap, interpolation=interpolation)
    ax.grid(None)
    ax.axis('off')
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    ax.set_frame_on(False)
    # pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #         hspace = 0, wspace = 0)
    # pyplot.gca().set_axis_off()
    # pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
    # pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())

    if save:
        pyplot.savefig(f"{randint(0, 10000):04x}.png", bbox_inches='tight', pad_inches = 0)

iimshow = lambda x: imshow(x, interpolation=None, cmap='hot')

def savefig(filename, format='png'):
    pyplot.savefig(filename, bbox_inches='tight', format=format)
