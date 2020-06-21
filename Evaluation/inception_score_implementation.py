# -*- coding: utf-8 -*-
"""Inception Score implementation.ipynb

"""

import numpy as np

def calculate_inception_score(p_yx,e=1*10^(-16)):
  p_y = np.expand_dims(p_yx.mean(axis=0), 0)
  kl_d = p_yx * (np.log(p_yx + e)) - np.log(p_y + e)
  sum_kl_d = kl_d.sum(axis=1)
  avg_kl_d = np.mean(sum_kl_d)
  score = np.exp(avg_kl_d)
  return score
