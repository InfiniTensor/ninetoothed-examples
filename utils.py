def replace_module(module, replacement_class):
    for child_name, child_module in module.named_children():
        if replacement_class.__name__ not in child_module.__class__.__name__:
            replace_module(child_module, replacement_class)
            continue

        replacement = replacement_class(child_module)
        setattr(module, child_name, replacement)
