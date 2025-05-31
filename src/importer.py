import sys
import os
import time
import h5py
import re
import random
import chardet
import pickle
import gc
import tracemalloc
import tempfile

from operator import itemgetter
from copy import deepcopy
from contextlib import redirect_stdout
from typing import Optional, Any

import collections.abc
from collections import defaultdict
from collections import Counter
from collections import deque

from itertools import chain
from tqdm import tqdm

import pandas as pd
import dask.dataframe as dd
import numpy as np
import scipy as sp
import networkx as nx

import sklearn as skl
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.python.keras.backend import dtype

from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from Levenshtein import distance, ratio

import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO

random.seed(123)
np.random.seed(123)
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(123)
tf.config.run_functions_eagerly(False)
tf.config.threading.set_inter_op_parallelism_threads(32)

print("Numpy Version: ", np.version.version)
print("Tensorflow Version: ", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print("Keras Version: ", tf.keras.__version__)