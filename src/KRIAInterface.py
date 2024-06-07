import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import QNN

class FConv2D_3x3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, weight, bias)

        result = QNN.quantize(F.conv2d(QNN.quantize(input), QNN.quantize(weight)))

        if bias is not None:
            reshapedBias = bias.reshape(1, bias.shape[0], 1, 1).repeat(result.shape[0], 1, result.shape[2], result.shape[3])
            result += reshapedBias

        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = F.conv_transpose2d(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input.transpose(0, 1), grad_output.transpose(0, 1)).transpose(0, 1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        # From https://discuss.pytorch.org/t/manual-backward-pass-of-standard-conv2d/120221
        #if ctx.needs_input_grad[0]:
        #    grad_input = torch.nn.grad torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        #if ctx.needs_input_grad[1]:
        #    grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        #if bias is not None and ctx.needs_input_grad[2]:
        #    grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias

class Conv2D_3x3(nn.Module):
    def __init__(self, inChannels, outChannels, dtype=torch.float32, bias=True):
        super(Conv2D_3x3, self).__init__()

        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.empty((outChannels, inChannels, 3, 3), dtype=dtype))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(outChannels, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters, from ConvNd.reset_parameters
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FConv2D_3x3.apply(input, self.weight, self.bias)