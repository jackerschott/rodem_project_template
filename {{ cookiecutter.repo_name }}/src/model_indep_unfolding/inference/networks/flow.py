from functools import partial
import math
import numpy as np
from numpy.typing import NDArray
import torch as T
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Optional, Iterable, Tuple, List, Callable

import FrEIA.framework as ff
import FrEIA.modules as fm

class ConditionalFlow(nn.Module):
    def __init__(self, inn_dim_in: int, cond_dim_in: int,
            inn_dim_subnet_internal: int, inn_num_blocks: int,
            inn_num_layers_per_block: int, spline_bound: int,
            permutation_seed: int) -> None:
        super().__init__()

        def create_fully_connected_net(dims_in, dims_out,
                dims_intern, num_layers):
            assert num_layers >= 2

            layers: List[nn.Module] = []

            layers.append(nn.Linear(dims_in, dims_intern))
            layers.append(nn.ReLU())

            for n in range(num_layers - 2):
                layers.append(nn.Linear(dims_intern, dims_intern))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(dims_intern, dims_out))

            return nn.Sequential(*layers)

        nodes = []

        node_condition = ff.ConditionNode(cond_dim_in)
        nodes.append(node_condition)

        nodes.append(ff.InputNode(inn_dim_in, 1, name='input'))
        nodes.append(ff.Node(nodes[-1], fm.Flatten, {}, name='flatten'))

        create_affine_coeff_net = partial(create_fully_connected_net,
                dims_intern=inn_dim_subnet_internal,
                num_layers=inn_num_layers_per_block)

        if permutation_seed is None:
            permutation_rng = np.random.default_rng()
        else:
            permutation_rng = np.random.default_rng(permutation_seed)

        for i in range(inn_num_blocks):
            perm = permutation_rng.permutation(inn_dim_in)
            args_coupling_block = dict(
                    subnet_constructor=create_affine_coeff_net,
                    perm=perm, bound=spline_bound)

            nodes.append(ff.Node(nodes[-1], SplineBlock,
                    args_coupling_block, conditions=node_condition,
                    name=f'spline_block_{i}'))
            self.block_type = SplineBlock

        nodes.append(ff.OutputNode(nodes[-1], name='output'))

        self.inn = ff.GraphINN(nodes)
        self.inn_dim_in = inn_dim_in
        self.cond_dim_in = cond_dim_in

    def forward(self, x: T.Tensor, c: T.Tensor, rev: bool = False,
            jac: bool = True) -> Tuple[T.Tensor, T.Tensor]:
        x_out, log_jacobian_det = self.inn(x, c, rev=rev, jac=jac)
        if rev:
            # FrEIA includes an extra dimension in this direction
            return x_out[..., 0], log_jacobian_det
        return x_out, log_jacobian_det

class SplineBlock(fm.InvertibleModule):
    BIN_WIDTH_MIN = 1e-3
    BIN_HEIGHT_MIN = 1e-3
    DELTA_MIN = 1e-3

    def __init__(self, dims_in: Iterable[Tuple[int]],
            dims_c: Iterable[Tuple[int]] = [],
            subnet_constructor: Optional[Callable] = None,
            perm: Optional[NDArray] = None, bin_count: int = 10,
            bound: float = 1) -> None:
        super().__init__(dims_in, dims_c)

        # only one dimensional data is supported (same for condition)
        assert len(dims_in) == 1 and len(dims_in[0]) == 1
        channel_count = dims_in[0][0]

        if len(dims_c) == 0:
            channel_count_cond = 0
            self.conditional = False
        else:
            assert len(dims_c[0]) == 1
            channel_count_cond = dims_c[0][0]
            self.conditional = True

        self.splits = (channel_count - math.ceil(channel_count / 2),
                math.ceil(channel_count / 2))

        # contrary to https://arxiv.org/abs/1906.04032, we have 3 * bin_count + 1
        # instead of 3 * bin_count - 1, because we also want to learn derivatives at
        # the boundary; we don't care about discontinuities if our data is (mostly)
        # in bounds, and otherwise you have to fix something anyway
        assert not subnet_constructor is None
        self.subnet = subnet_constructor(self.splits[0] + channel_count_cond,
                (3 * bin_count + 1) * self.splits[1])

        if perm is None:
            perm = np.random.permutation(channel_count)

        channel_perm = np.zeros((channel_count, channel_count))
        for i, j in enumerate(perm):
            channel_perm[i, j] = 1.

        channel_perm = T.FloatTensor(channel_perm)
        self.channel_perm = nn.Parameter(channel_perm, requires_grad=False)
        channel_inv_perm = T.FloatTensor(channel_perm.T)
        self.channel_inv_perm = nn.Parameter(channel_inv_perm, requires_grad=False)

        self.bin_count = bin_count
        self.bound = bound


    def forward(self, x: T.Tensor, c: Iterable[T.Tensor] = [], rev=False, jac=True):
        x, = x

        if rev:
            x = nnf.linear(x, self.channel_inv_perm)

        x1, x2 = T.split(x, self.splits, dim=-1)

        if self.conditional:
            x1_with_cond = T.cat([x1, *c], dim=1)
        else:
            x1_with_cond = x1

        # we want to compute theta for each channel of the second split, but the
        # output of the subnet will be a flattened version of theta, cramming
        # together the channel and parameter dimension of theta; so reshape
        theta = self.subnet(x1_with_cond)
        theta = theta.reshape(-1, self.splits[1], 3 * self.bin_count + 1)

        x2, log_jac_det_2 = self._apply_spline(x2, theta, inverse=rev)

        log_jac_det = log_jac_det_2
        x_out = T.cat((x1, x2), dim=1)

        if not rev:
            x_out = nnf.linear(x_out, self.channel_perm)

        return (x_out,), log_jac_det

    def _apply_spline(self, x: T.Tensor, theta: T.Tensor, inverse=False) \
            -> Tuple[T.Tensor, T.Tensor]:
        x_knots, y_knots, bin_widths, bin_heights, knot_slopes \
            = self._unpack_spline_params(theta)
        x_out, log_jac_det = spline(x, x_knots, y_knots,
                bin_widths, bin_heights, knot_slopes, inverse=inverse)
        return x_out, log_jac_det

    def _unpack_spline_params(self, theta: T.Tensor) \
            -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        bin_widths_offset, bin_heights_offset, knot_slopes_offset \
            = 0, self.bin_count, 2 * self.bin_count

        bin_widths_unconstr = theta[..., bin_widths_offset : bin_heights_offset]
        bin_heights_unconstr = theta[..., bin_heights_offset : knot_slopes_offset]
        knot_slopes_unconstr = theta[..., knot_slopes_offset:]

        bin_widths = 2 * self.bound * self._softmax_with_min(
                    bin_widths_unconstr, min=self.BIN_HEIGHT_MIN)
        bin_heights = 2 * self.bound * self._softmax_with_min(
                    bin_heights_unconstr, min=self.BIN_WIDTH_MIN)
        knot_slopes = self._softplus_with_min(
                knot_slopes_unconstr, min=self.DELTA_MIN)

        # cumsum starts at bin_widths[..., 0], but we want to start at zero
        x_knots = nnf.pad(T.cumsum(bin_widths, dim=-1),
                (1, 0), value=0) - self.bound
        y_knots = nnf.pad(T.cumsum(bin_heights, dim=-1),
                (1, 0), value=0) - self.bound
        
        return x_knots, y_knots, bin_widths, bin_heights, knot_slopes


    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return input_dims


    def _softplus_with_min(self, x: T.Tensor, min=1e-3) -> T.Tensor:
        # give one when x is zero, give min when softplus(x) is zero
        return min + (1 - min) * nnf.softplus(x) / math.log(2)
    def _softmax_with_min(self, x: T.Tensor, min=1e-3) -> T.Tensor:
        # give min when any softmax(x) is zero but still sum to one
        return min + (1 - min * x.shape[-1]) * nnf.softmax(x, dim=-1)

def spline(xy: T.Tensor, x_knots: T.Tensor, y_knots: T.Tensor, bin_widths: T.Tensor,
        bin_heights: T.Tensor, knot_slopes: T.Tensor, inverse=False) -> Tuple[T.Tensor, T.Tensor]:
    bin_count = bin_widths.shape[-1]

    xy_knots = (x_knots, y_knots)[inverse]
    # make x contiguous here (it becomes non-contiguous from the split in
    # forward), otherwise we get a performance warning
    bin_indices = T.searchsorted(xy_knots,
            xy[..., None].contiguous()).squeeze(-1)
    # searchsorted returns the index of the knot after the value of xy, but
    # we want the knot before xy
    bin_indices = bin_indices - 1

    in_bounds = (bin_indices >= 0) & (bin_indices < bin_count)
    bin_indices_in_bounds = bin_indices[in_bounds]

    x1 = x_knots[in_bounds, bin_indices_in_bounds]
    y1 = y_knots[in_bounds, bin_indices_in_bounds]
    x_width = bin_widths[in_bounds, bin_indices_in_bounds]
    y_width = bin_heights[in_bounds, bin_indices_in_bounds]
    delta_1 = knot_slopes[in_bounds, bin_indices_in_bounds]
    delta_2 = knot_slopes[in_bounds, bin_indices_in_bounds + 1]

    x_left_bound = x_knots[~in_bounds, [0]]
    x_right_bound = x_knots[~in_bounds, [-1]]
    y_bottom_bound = y_knots[~in_bounds, [0]]
    y_top_bound = y_knots[~in_bounds, [-1]]

    yx = T.full_like(xy, np.nan)
    log_jac_diag = T.full_like(xy, 0) # jac_diag is one when out of bounds
    yx[~in_bounds] = linear(xy[~in_bounds], x_left_bound, x_right_bound,
            y_bottom_bound, y_top_bound, inverse=inverse)
    yx[in_bounds], log_jac_diag[in_bounds] = rational_quadratic(xy[in_bounds],
            x1, y1, x_width, y_width, delta_1, delta_2, inverse=inverse)
    log_jac_det = T.sum(log_jac_diag, dim=-1)

    return yx, log_jac_det

def linear(xy: T.Tensor, x1: T.Tensor, x2: T.Tensor,
        y1: T.Tensor, y2: T.Tensor, inverse=False) -> T.Tensor:
    if not inverse:
        x = xy
        return (y2 - y1) / (x2 - x1) * (x - x1) + y1
    else:
        y = xy
        return (x2 - x1) / (y2 - y1) * (y - y1) + x1

def rational_quadratic(xy: T.Tensor, x1: T.Tensor, y1: T.Tensor,
        x_width: T.Tensor, y_width: T.Tensor, delta_1: T.Tensor, delta_2: T.Tensor, inverse=False):
    # Compute rational quadratic spline between (x1, y1) and (x1 + x_width,
    # y1 + y_width) with derivatives delta_1 and delta_2 at these points, or compute
    # the inverse; compare https://arxiv.org/abs/1906.04032, eqs. (4)-(8)
    s = y_width / x_width

    if not inverse:
        x = xy

        eta = (x - x1) / x_width
        rev_eta = 1 - eta
        eta_rev_eta = eta * rev_eta

        y_denom = (s + (delta_1 + delta_2 - 2 * s) * eta_rev_eta)
        y = y1 + y_width * (s * eta**2 + delta_1 * eta_rev_eta) / y_denom
        yx = y

        # compute jacobian
        jac_numer = s**2 * (delta_2 * eta**2
                + 2 * s * eta_rev_eta + delta_1 * rev_eta**2)

        # jac_diag = jac_diag_numer / jac_diag_denom,
        # where jac_diag_denom = y_denom**2
        log_jac = T.log(jac_numer) - 2 * T.log(y_denom)
    else:
        y = xy

        y_shifted = y - y1
        shifted_delta_sum = (delta_2 + delta_1 - 2 * s)

        a = y_width * (s - delta_1) + y_shifted * shifted_delta_sum
        b = y_width * delta_1 - y_shifted * shifted_delta_sum
        c = -s * y_shifted

        eta = 2 * c / (-b - T.sqrt(b**2 - 4 * a * c))
        rev_eta = 1 - eta
        eta_rev_eta = eta * rev_eta

        x = x1 + x_width * eta
        yx = x

        # compute jacobian identically to non-inverse case, just with different
        # optimisations
        inv_jac_numer = s**2 * (delta_2 * eta**2
                + 2 * s * eta_rev_eta + delta_1 * rev_eta**2)
        inv_jac_denom = (s + shifted_delta_sum * eta_rev_eta)**2

        log_inv_jac = T.log(inv_jac_numer) - T.log(inv_jac_denom)
        log_jac = -log_inv_jac

    return yx, log_jac
