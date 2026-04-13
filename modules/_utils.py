from contextlib import contextmanager


def replace_module(module, replacement_class):
    """Recursively replace modules whose class name contains the replacement class name."""

    for child_name, child_module in module.named_children():
        if replacement_class.__name__ not in child_module.__class__.__name__:
            replace_module(child_module, replacement_class)

            continue

        replacement = replacement_class(child_module)
        setattr(module, child_name, replacement)


def _make_backend_manager(cls, attr, impls):
    """Create a context manager that switches a class attribute to a different backend.

    :param cls: The module class whose attribute will be swapped.
    :param attr: The name of the class attribute to swap.
    :param impls: Dict mapping backend name to callable implementation.
    :return: A context manager function.
    """

    @contextmanager
    def backend(backend_name):
        prev = getattr(cls, attr)
        setattr(cls, attr, _get_impl(backend_name, impls))

        try:
            yield
        finally:
            setattr(cls, attr, prev)

    return backend


def _get_impl(backend_name, impls):
    """Return the implementation for the given backend name."""

    if backend_name not in impls:
        raise ValueError(f"Unknown backend: `{backend_name}`.")

    return impls[backend_name]
