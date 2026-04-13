import torch
import triton

DISPLAY_NAMES = {
    "ninetoothed": "NineToothed",
    "torch": "PyTorch",
    "triton": "Triton",
}

STYLES = [
    ("blue", "-"),
    ("green", "-"),
    ("orange", "-"),
    ("red", "-"),
    ("purple", "-"),
    ("cyan", "-"),
]


def assert_match(impls, args, kwargs=None, tolerances=None):
    """Assert that all implementations produce matching outputs.

    Same API as ``check``, but raises ``AssertionError`` on mismatch
    instead of printing. Intended for use in test suites.

    :param impls: Ordered dict mapping provider name to callable.
    :param args: Tuple of positional arguments.
    :param kwargs: Dict of keyword arguments.
    :param tolerances: Dict mapping provider name to ``torch.allclose`` kwargs.
    """
    kwargs = kwargs or {}
    tolerances = tolerances or {}
    results = {name: fn(*args, **kwargs) for name, fn in impls.items()}

    names = list(impls)
    reference_name = names[0]
    reference = results[reference_name]

    for name in names[1:]:
        tol = tolerances.get(name, {})
        ref_display = _display_name(reference_name)
        other_display = _display_name(name)

        assert torch.allclose(reference, results[name], **tol), (
            f"{ref_display} and {other_display} outputs differ."
        )


def benchmark(
    impls,
    make_inputs,
    x_names,
    x_vals,
    name,
    benchmark_args=None,
    x_log=True,
    assert_correctness=True,
    tolerances=None,
    save_path=".",
):
    """Create and run a performance benchmark.

    :param impls: Ordered dict mapping provider name to callable.
    :param make_inputs: Callable returning ``(args_tuple, kwargs_dict)``.
    :param x_names: List of benchmark parameter names.
    :param x_vals: List of benchmark parameter values.
    :param name: Operator name, used for the plot filename.
    :param benchmark_args: Fixed benchmark args dict.
    :param x_log: Whether to use log scale for the x-axis.
    :param tolerances: Dict mapping provider name to ``torch.allclose`` kwargs.
    :param assert_correctness: Whether to assert correctness at each point.
    :param save_path: Directory to save plot files, or ``None`` to skip saving.
    """
    providers = list(impls)
    tolerances = tolerances or {}

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals,
            line_arg="provider",
            line_vals=providers,
            line_names=[_display_name(p) for p in providers],
            plot_name=f"{name}-performance",
            args=benchmark_args or {},
            ylabel="ms",
            x_log=x_log,
            styles=[_style(i) for i in range(len(providers))],
        )
    )
    def bench(provider, **params):
        args, kwargs = make_inputs(**params)

        if assert_correctness:
            results = {p: impls[p](*args, **kwargs) for p in providers}
            reference = results[providers[0]]

            for p in providers[1:]:
                tol = tolerances.get(p, {})
                assert torch.allclose(reference, results[p], **tol)

        return triton.testing.do_bench(lambda: impls[provider](*args, **kwargs))

    bench.run(print_data=True, save_path=save_path)


def _display_name(name):
    """Return the display name for a provider."""

    return DISPLAY_NAMES.get(name, name)


def _style(index):
    """Return a plot style, cycling through available options."""

    return STYLES[index % len(STYLES)]
