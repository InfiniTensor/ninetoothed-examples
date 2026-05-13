import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

from ops.ninetoothed.kernels._common import build


def arrangement(q, k, v, scale, q_start, o, BLOCK_SIZE_M, BLOCK_SIZE_N):
    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_k_or_v(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_N, -1))
        arranged = arranged.tile((1, 1, -1, -1))
        arranged = arranged.expand((-1, -1, q_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    q_arranged = arrange_q_or_o(q)

    return (
        q_arranged,
        arrange_k_or_v(k),
        arrange_k_or_v(v),
        scale,
        q_start,
        arrange_q_or_o(o),
    )


def application(q, k, v, scale, q_start, o):
    q_loaded = (q * scale * 1.44269504089).to(q.dtype)

    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(
            (q.offsets(-2) + q_start)[:, None] >= k[i].offsets(-2),
            qk,
            float("-inf"),
        )

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v.dtype.dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc.to(o.dtype)  # noqa: F841


def premake(head_dim, dtype, block_size_m, block_size_n):
    arrangement_ = functools.partial(
        arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
    )
    shape = (None, None, None, head_dim)
    tensors = (
        Tensor(shape=shape, dtype=dtype),
        Tensor(shape=shape, dtype=dtype),
        Tensor(shape=shape, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(shape=shape, dtype=dtype),
    )

    return arrangement_, application, tensors


def _configs(head_dim, dtype):
    return tuple(
        (
            (),
            {
                "head_dim": head_dim,
                "dtype": dtype,
                "block_size_m": bm,
                "block_size_n": bn,
            },
            {},
        )
        for bm in (32, 64, 128, 256, 512)
        for bn in (32, 64, 128, 256, 512)
        if bm * head_dim <= 32768 and bn * head_dim <= 32768
    )


@functools.cache
def _kernel(head_dim, dtype):
    return build(
        premake,
        _configs(head_dim, dtype),
        meta_parameters=("block_size_m", "block_size_n"),
        kernel_name=f"scaled_dot_product_attention_{head_dim}",
    )


def kernel(q, k, v, scale, q_start, o, head_dim, dtype):
    return _kernel(head_dim, dtype)(q, k, v, scale, q_start, o, head_dim, dtype)
