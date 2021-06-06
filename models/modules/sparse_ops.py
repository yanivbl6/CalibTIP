import torch
from torch import autograd, nn
import torch.nn.functional as F

from itertools import repeat
from torch._six import container_abcs
import numpy as np

class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, P, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]
        F_indexes= index[:, :int(M-N-P)]

        w_f = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_f = w_f.scatter_(dim=1, index=F_indexes, value=0).reshape(weight.shape)

        if P > 0:
            P2_indexes= index[:, int(M-N-P):]
            w_p = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_p = w_f.scatter_(dim=1, index=P2_indexes, value=0).reshape(weight.shape)

            ctx.mask = w_f + w_p
            tmp1 = output*w_f
            tmp2 = round_to_power_of_2(output*w_p)
            w_out = tmp1 + tmp2
        else:
            ctx.mask = w_f
            w_out = output*w_f

        ctx.decay = decay

        return w_out


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None, None


def round_to_power_of_2(w):
    return torch.pow(2.0, (w.abs().log()/np.log(2.0)).floor()) * w.sign()

def block_quant(weight, block, num_bits, second_sort , M, N, decay = 0.0002, lsb = True):
    mod_factor = int(2**(num_bits - second_sort))
    length = weight.numel()
    group = int(length/M)

    ##import pdb; pdb.set_trace()


    if lsb == 0:
        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(-(weight_temp % mod_factor) , dim=1)
    elif (lsb == 1):
        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(-(weight_temp) , dim=1)
    else:
        weight_temp = weight.detach().permute(0,2,3,1).reshape(group, M)
        #bitmask = torch.ones(weight_temp.shape, dtype = torch.int8, device=weight_temp.device)* lsb
        index = torch.argsort(-(weight_temp.bitwise_and(~lsb)) , dim=1)

    index_pool = torch.zeros(weight_temp.shape, device=weight_temp.device, dtype = torch.int8)
    
    w_out = 0
    i = 0
    for bits, count in block.items():
        if bits <= 0:
            break
        
        P_indexes= index[:, i:i+count]
        w_p = index_pool.scatter(dim=1, index=P_indexes, value=1).reshape(weight.permute(0,2,3,1).shape)
        w_p = w_p.permute(0,3,1,2)

        if (lsb==0):
            reduceFactor = int(2**(num_bits-bits))
            tmp =  (weight*w_p // reduceFactor) * reduceFactor
        elif (lsb == 1):
            minval = -int(2**(bits-1))
            maxval = minval + int(2**(bits)) -1
            tmp =  (weight*w_p).clamp(minval,maxval)
        else:
            ##bitmask = torch.ones(weight_temp.shape, dtype = torch.int8, device=weight_temp.device)* lsb
            if bits == num_bits:
                tmp =  weight*w_p
            else:
                tmp =  (weight*w_p).bitwise_and(lsb)

        w_out = w_out + tmp

        i = i + count

    return w_out

def block_quant2(weight, block, num_bits, second_sort , M, N, decay = 0.0002, lsb = True):
    mod_factor = int(2**(num_bits - second_sort))
    length = weight.numel()
    group = int(length/M)

    ##import pdb; pdb.set_trace()


    if lsb == 0:
        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(-(weight_temp % mod_factor) , dim=1)
    elif (lsb == 1):
        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(-(weight_temp) , dim=1)
    else:
        weight_temp = weight.detach().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(-(weight_temp.bitwise_and(~lsb)) , dim=1)
    
    index_pool = torch.zeros(weight_temp.shape, device=weight_temp.device, dtype = torch.int8)
    
    w_out = 0
    i = 0
    

    block.keys

    bits = num_bits
    count = block[bits]

    block[0] = block[0] + count

    P_indexes= index[:, i:i+count]
    w_p = index_pool.scatter_(dim=1, index=P_indexes, value=1).reshape(weight.permute(0,2,3,1).shape)
    w_p = w_p.permute(0,3,1,2)

    if (lsb==0):
        reduceFactor = int(2**(num_bits-bits))
        tmp =  (weight*w_p // reduceFactor) * reduceFactor
    elif (lsb == 1):
        minval = -int(2**(bits-1))
        maxval = minval + int(2**(bits)) -1
        tmp =  (weight*w_p).clamp(minval,maxval)
    else:
        if bits == num_bits:
            tmp =  weight*w_p
        else:
            tmp =  (weight*w_p).bitwise_and(lsb)


    w_out = w_out + tmp

    i = i + count

    return w_out

class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, block, num_bits, second_sort, M, P, N, lsb = np.int8(0b11111000), decay = 0.0002):

        assert(M >= N+P)
        assert(N >= 0)
        assert(P >= 0)

        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        if N>0:
            weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)

            index = torch.argsort(weight_temp, dim=1)[:, :N]
            F_indexes= index[:, :N]

            w_f = torch.ones(weight_temp.shape, device=weight_temp.device, dtype= torch.int8)
            w_f = w_f.scatter_(dim=1, index=F_indexes, value=0).reshape(weight.permute(0,2,3,1).shape)
            w_f = w_f.permute(0,3,1,2)

            ctx.mask = w_f
            w_out = output*w_f
        else:
            ctx.mask = torch.ones(output.shape)
            w_out = output


        if P > 0:
            w_out = block_quant(w_out, block, num_bits, second_sort, M, N, decay, lsb)
        
        ctx.decay = decay

        return w_out

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None, None



    


