def replace_module(module, replacement_class):
    for child_name, child_module in module.named_children():
        if replacement_class.__name__ not in child_module.__class__.__name__:
            replace_module(child_module, replacement_class)
            continue

        replacement = replacement_class(child_module)
        setattr(module, child_name, replacement)


def find_module_types(module):
    types = {type(module)}

    for child_module in module.children():
        types.update(find_module_types(child_module))

    return types
