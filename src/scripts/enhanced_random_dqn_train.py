#!/usr/bin/env python3
"""
Enhanced Randomized Radii Parallel Training
==========================================
Small improvements over random_dqn_train.py for better performance.
"""

import sys
import os
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.algorithms.dqn_agent import (
    AdvancedCirclePlacementEnv,
    GuidedDQNAgent,
    random_seeder,
    compute_included,
    HeatmapFeatureExtractor,
)
from src.utils.periodic_tracker import PeriodicTaskTracker, RobustPeriodicChecker