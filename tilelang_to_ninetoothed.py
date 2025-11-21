import ast
import inspect
import textwrap

import ninetoothed
import ninetoothed.language as ntl
import tilelang
import tilelang.language as T
import triton
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, cache_source
from ninetoothed.jit import import_from_path


class _TileLangToNineToothedTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

        self._variables = {}

        self._num_warps = None

        self._warp_size = 32

    def __call__(self, func):
        func_def = ast.parse(textwrap.dedent(inspect.getsource(func)))

        self.visit(func_def)

        def _arrangement(*_args):
            return tuple(self._variables[f"{arg}_arranged"] for arg in self._args)

        source_file = str(cache_source(self._application_source))

        module = import_from_path(source_file, source_file)
        module_vars = vars(module)

        application = module_vars[type(self)._APPLICATION_NAME]

        tensors = tuple(self._variables[arg] for arg in self._args)

        kernel = ninetoothed.make(_arrangement, application, tensors)

        return kernel

    def visit_FunctionDef(self, node):
        decorators = set(
            eval(ast.unparse(decorator)) for decorator in node.decorator_list
        )

        def _visit_tilelang_jit():
            for arg in node.args.args:
                self._variables[arg.arg] = Symbol(arg.arg, constexpr=True)

        def _visit_t_prim_func():
            self._args = tuple(arg.arg for arg in node.args.args)

            for arg in node.args.args:
                shape = eval(
                    ast.unparse(arg.annotation.args[0]), globals(), self._variables
                )

                if len(arg.annotation.args) > 1:
                    dtype = eval(
                        ast.unparse(arg.annotation.args[1]), globals(), self._variables
                    )
                else:
                    dtype = ninetoothed.float32

                self._variables[arg.arg] = Tensor(shape=shape, dtype=dtype)

        if tilelang.jit in decorators:
            _visit_tilelang_jit()
        elif T.prim_func in decorators:
            _visit_t_prim_func()

        self.generic_visit(node)

        return node

    def visit_With(self, node):
        for item in node.items:
            if eval(ast.unparse(item.context_expr.func)) is not T.Kernel:
                return node

        self.generic_visit(node)

        node.body = [
            type(self)._KernelBodyTransformer(self._variables, self._grid).visit(stmt)
            for stmt in node.body
        ]

        name = type(self)._APPLICATION_NAME
        params = ", ".join(self._args)
        body = textwrap.indent(ast.unparse(node.body), type(self)._INDENT)

        self._application_source = f"def {name}({params}):\n{body}"

        return node

    def visit_withitem(self, node):
        class _MathFunctionTransformer(ast.NodeTransformer):
            def visit_Call(self, node):
                if eval(ast.unparse(node.func)) is T.ceildiv:
                    node.func = Symbol(f"{triton.__name__}.{triton.cdiv.__name__}").node

                self.generic_visit(node)

                return node

        context_expr = _MathFunctionTransformer().visit(node.context_expr)

        self._grid = tuple(
            eval(ast.unparse(arg), globals(), self._variables)
            for arg in context_expr.args
        )

        for keyword in context_expr.keywords:
            if keyword.arg != "threads":
                continue

            self._num_warps = (
                eval(ast.unparse(keyword.value), globals(), self._variables)
                // self._warp_size
            )

        if isinstance(node.optional_vars, ast.Tuple):
            for dim, element in enumerate(node.optional_vars.elts):
                if not isinstance(element, ast.Name):
                    continue

                self._variables[element.id] = CodeGenerator._name_for_index(
                    Tensor(shape=self._grid, source=self._variables[self._args[0]]), dim
                )

        self.generic_visit(node)

        return node

    _INDENT = "    "

    _APPLICATION_NAME = "application"

    class _KernelBodyTransformer(ast.NodeTransformer):
        def __init__(self, variables, grid):
            super().__init__()

            self._variables = variables

            self._grid = grid

        def visit_Assign(self, node):
            self.generic_visit(node)

            assert (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            ), "Currently, only the simplest form of assignment is supported."

            name = node.targets[0].id
            shape = eval(ast.unparse(node.value.args[0]), globals(), self._variables)
            dtype = eval(
                ast.unparse(node.value.keywords[0].value), globals(), self._variables
            )

            self._variables[name] = Tensor(shape=shape, dtype=dtype)

            return node

        def visit_Call(self, node):
            func = eval(ast.unparse(node.func))

            self.generic_visit(node)

            if func in (T.alloc_shared, T.alloc_fragment):
                node.func = ntl.attribute("zeros").node
                node.keywords = [ast.keyword(arg="dtype", value=node.args[1])]
                node.args.pop(1)
            elif func is T.clear:
                name = node.args[0].id

                return ast.parse(
                    f"{name} = {ntl.call('zeros', f'{name}.shape', dtype=f'{name}.dtype')}"
                ).body[0]
            elif func is T.Pipelined:
                node.func = ntl.attribute("range").node
            elif func is T.ceildiv:
                node.func = ntl.attribute("cdiv").node
            elif func is T.copy:
                src = node.args[0]
                dst = node.args[1]

                if isinstance(dst, ast.Name):
                    tensor_name = src.value.id

                    tensor = self._variables[tensor_name]
                    tile = self._variables[dst.id]
                elif isinstance(src, ast.Name):
                    tensor_name = dst.value.id

                    tensor = self._variables[tensor_name]
                    tile = self._variables[src.id]

                self._variables[f"{tensor_name}_arranged"] = (
                    tensor.tile(tile.shape, strides=(1, 1))
                    .tile((-1, -1))
                    .expand(self._grid)
                )

                if isinstance(dst, ast.Name):
                    return ast.parse(f"{dst.id} = {ast.unparse(src)}").body[0]
                elif isinstance(src, ast.Name):
                    return ast.parse(f"{ast.unparse(dst)} = {src.id}").body[0]
            elif func is T.gemm:
                a, b, c = (node.args[i].id for i in range(3))

                return ast.parse(f"{c} += {ntl.call('dot', a, b)}").body[0]

            return node

        def visit_Constant(self, node):
            # TODO: Use `ntl` instead of `tl` here.
            if node.value == "float":
                return Symbol("triton.language.float32").node

            if node.value == "float16":
                return Symbol("triton.language.float16").node

            return node

        def visit_Name(self, node):
            if node.id in self._variables:
                if isinstance(variable := self._variables[node.id], Symbol):
                    self._variables[str(variable)] = variable

                    return variable.node

            return node


transform_tilelang_to_ninetoothed = _TileLangToNineToothedTransformer()
