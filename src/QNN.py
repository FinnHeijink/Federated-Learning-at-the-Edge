#
# Charlotte Frenkel, Institute of Neuroinformatics, 2021-2022
#
# Credit for the surrogate gradient function: F. Zenke, https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
#

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.autograd import Function


def quant_one(W, nb):
    if nb == 1:
        return (W >= 0).float() * 2 - 1
    elif nb == 2:
        return torch.clamp(torch.round(W), -1, 1)
    non_sign_bits = nb - 1
    m = pow(2, non_sign_bits)
    return torch.clamp(torch.round(W * m), -m, m - 1) / m

def quant_fraction(W, nb, nfract):
    if nb == 1:
        return (W >= 0).float() * 2 - 1
    elif nb == 2:
        return torch.clamp(torch.round(W), -1, 1)

    non_sign_bits = nb - 1

    f = pow(2, nfract)
    m = pow(2, non_sign_bits)

    return torch.clamp(torch.round(W * f), -m, m - 1) / f

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class QuantizeTensor(torch.autograd.Function):
    nb = 8
    nf = 4

    @staticmethod
    def forward(ctx, input):
        return quant_fraction(input, QuantizeTensor.nb, QuantizeTensor.nf)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class QuantizeModel(nn.Module):
    def forward(self, input):
        return quantize(input)

spikeAct = SurrGradSpike.apply
quantize = QuantizeTensor.apply

