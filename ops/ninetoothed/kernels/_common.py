import ninetoothed
import ninetoothed.generation

DTYPES = (ninetoothed.float16, ninetoothed.bfloat16)


def build(premake, configs, *, meta_parameters=None, kernel_name):
    output_dir = ninetoothed.generation.CACHE_DIR / "examples" / kernel_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return ninetoothed.build(
        premake,
        configs,
        meta_parameters=meta_parameters,
        kernel_name=kernel_name,
        output_dir=output_dir,
        lazy=True,
    )
