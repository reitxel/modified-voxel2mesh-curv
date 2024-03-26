import torch.nn.functional as F
import torch

from IPython import embed 
import time
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

MODE_ZEROS = 0
MODE_BORDER = 1


def affine_grid(theta, size):
    return AffineGridGenerator.apply(theta, size)


# TODO: Port these completely into C++
class AffineGridGenerator(Function):

    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size
        N, C, D, H, W = size
        ctx.size = size

        #ctx.is_cuda = True

        base_grid = theta.new(N, D, H, W, 4)

        w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
        h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
        d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

        base_grid[:, :, :, :, 0] = w_points
        base_grid[:, :, :, :, 1] = h_points
        base_grid[:, :, :, :, 2] = d_points
        base_grid[:, :, :, :, 3] = 1
        ctx.base_grid = base_grid
        grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
        grid = grid.view(N, D, H, W, 3)
        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        N, C, D, H, W = ctx.size
        assert grad_grid.size() == torch.Size([N, D, H, W, 3])
        base_grid = ctx.base_grid
        grad_theta = torch.bmm(
            base_grid.view(N, D * H * W, 4).transpose(1, 2),
            grad_grid.view(N, D * H * W, 3))
        grad_theta = grad_theta.transpose(1, 2)
        return grad_theta, None


def stn_all_ratations(params, inverse=False):
    theta, theta_x, theta_y, theta_z = stn_all_ratations_with_all_theta(params, inverse)
    return theta

def stn_quaternion_rotations(params):

    params = params.view(3)
    qi, qj, qk = params

    s = qi ** 2 + qj ** 2 + qk ** 2

    theta = torch.eye(4, device=params.device)


    theta[0, 0] = 1 - 2 * s * (qj ** 2 + qk ** 2)
    theta[1, 1] = 1 - 2 * s * (qi ** 2 + qk ** 2)
    theta[2, 2] = 1 - 2 * s * (qi ** 2 + qj ** 2)

    theta[0, 1] = 2 * s * qi * qj
    theta[0, 2] = 2 * s * qi * qk

    theta[1, 0] = 2 * s * qi * qj
    theta[1, 2] = 2 * s * qj * qk

    theta[2, 0] = 2 * s * qi * qk
    theta[2, 1] = 2 * s * qj * qk

    return theta

 

def stn_batch_quaternion_rotations(params, inverse=False):
    thetas = []
    for param in params:
        theta = stn_quaternion_rotations(param)
        # if inverse:
        #     theta = theta.inverse()
        thetas.append(theta)

    thetas = torch.cat(thetas, dim=0)
    thetas = thetas.view(-1,4,4)
    return thetas

def scale(param):
    theta_scale = torch.eye(4) 

    theta_scale[0, 0] = param
    theta_scale[1, 1] = param
    theta_scale[2, 2] = param

    return theta_scale

def rotate(angles):
    angle_x, angle_y, angle_z = angles
    params = torch.Tensor([torch.cos(angle_x), torch.sin(angle_x), torch.cos(angle_y), torch.sin(angle_y),torch.cos(angle_z), torch.sin(angle_z)])
    params = params.view(3,2)
    theta = stn_all_ratations(params)

    return theta
 
def shift(axes):
    theta = torch.eye(4, device=axes.device)
    theta[0, 3] = axes[0]
    theta[1, 3] = axes[1]
    theta[2, 3] = axes[2]

    return theta

def transform(theta, x, y=None, w=None, w2=None):
    theta = theta[0:3, :].view(-1, 3, 4)
    grid = affine_grid(theta, x[None].shape)
    if x.device.type == 'cuda':
        grid = grid.cuda()
    x = F.grid_sample(x[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0]
    if y is not None:
        y = F.grid_sample(y[None, None].float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()[0, 0]
    else:
        return x 
    if w is not None: 
        w = F.grid_sample(w[None, None].float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()[0, 0] 
        return x, y, w
    else:
        return x, y
    # if w2 is not None:
        # w2 = F.grid_sample(w2[None, None].float(), grid, mode='nearest', padding_mode='zeros').long()[0, 0]
    


