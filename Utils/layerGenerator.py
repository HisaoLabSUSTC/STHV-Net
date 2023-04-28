# Function to get a "Transform Layer"
# with a linear layer and a possible batchNorm layer included
# for the model
import torch
from torch import nn

from setTransformer.models import SetTransformer


def getTransformLayer(forwardDim):
    transformLayer = nn.Sequential(
        nn.Linear(forwardDim, forwardDim)
    )
    return transformLayer


def getActivationLayer():
    activationLayer = nn.Sequential(
        nn.ReLU()
    )
    return activationLayer
