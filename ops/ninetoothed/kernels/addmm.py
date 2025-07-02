import ninetoothed
from ninetoothed import Tensor

import ops.ninetoothed.kernels.mm as mm


def arrangement(input, mat1, mat2, beta, alpha, output):
    _, _, input_arranged = mm.arrangement(mat1, mat2, input)

    mat1_arranged, mat2_arranged, output_arranged = mm.arrangement(mat1, mat2, output)

    return input_arranged, mat1_arranged, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    mm.application(mat1, mat2, output)
    output = beta * input + alpha * output


tensors = (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
