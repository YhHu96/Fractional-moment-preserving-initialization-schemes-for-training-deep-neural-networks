# -*- coding: utf-8 -*-

from __future__ import division
from scipy.special import digamma
import torch
import torch.nn as nn
import math
import numpy as np
def yuanhan_normal_(tensor, a=0.01, mode='fan_in', nonlinearity='leaky_relu', s=1):

    # According to the paper Fractional moment-preserving initialization schemes for training deep neural networks
    # s is the order of moment that is perserved

    fan = nn.init._calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    if nonlinearity == 'relu':
        std = math.sqrt(2/fan+5*(2-s)/(2*fan**2))
    elif nonlinearity == 'linear':
        std = math.sqrt(1/fan+(2-s)/(2*fan**2))
    else:
        m = fan
        temp = 2/(1+a*a)/m+(2-s)*((5*s-24)*a*a+10)/(2*(s+2)*a*a+4)/m/m
        std = math.sqrt(temp)
    with torch.no_grad():
        return tensor.normal_(0, std)

def randomwalk_normal_(tensor,a=0.01,mode='fan_in',nonlinearity='leaky_relu'):

    # New implemented randomwalk initialization scheme
    
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    if gain == 1:
        std = gain*np.exp(1/(2*fan))/np.sqrt(fan)
    else:
        std = gain*np.exp(1.2/(fan-2.4))/np.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0,std)

yuanhan_normal = nn.init._make_deprecate(yuanhan_normal_)