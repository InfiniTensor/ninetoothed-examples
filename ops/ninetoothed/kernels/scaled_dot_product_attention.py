import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()


def arrangement(
    q, k, v, scale, q_start, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
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


_shape_options = (None, None, None, {"constexpr": True, "upper_bound": 128})
_q, _k, _v, _o = (Tensor(4, shape_options=_shape_options) for _ in range(4))
tensors = (_q, _k, _v, Tensor(0), Tensor(0), _o)

kernel = ninetoothed.make(arrangement, application, tensors)
