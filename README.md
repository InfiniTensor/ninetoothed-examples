# NineToothed Examples

This repository contains examples for [NineToothed](https://github.com/InfiniTensor/ninetoothed), including implementations of several common compute kernels written using NineToothed.

## Usage

After cloning this repository, you can run any of the examples using Python. For instance, to run the matrix multiplication example, execute the following command:

```bash
python matmul.py
```

### Autotuning Behavior

By default, the examples apply autotuning, which may take several minutes or longer to complete for complex kernels. If you wish to disable autotuning, you can replace symbol definitions with concrete values. Consider the following example:

```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)
```

Here, `meta=True` specifies that `BLOCK_SIZE` is a meta symbol for autotuning. To disable autotuning, you can:

1. Set `constexpr=True` and pass a value when invoking the kernel.
2. Replace the symbol definition with a fixed integer value, as shown below:

```python
BLOCK_SIZE = 1024
```

These approaches allow you to obtain results in seconds. However, selecting optimal values is crucial for good performance. Experiment with different values to determine the best configuration.

## Third-Party Code and Licenses

This project includes code modified or inspired from the following open-source repositories:

* [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
* [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)
* [https://github.com/ROCm/triton](https://github.com/ROCm/triton)
* [https://github.com/l1351868270/implicit_gemm.triton](https://github.com/l1351868270/implicit_gemm.triton)

Licenses for third-party code are stored in the `third_party` directory. Each subdirectory contains its associated `LICENSE` file.

## License

This repository is distributed under the Apache-2.0 license. See the included [LICENSE](LICENSE) file for details.
