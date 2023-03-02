def rst_comment():
    text = "This file has been auto-generated. DO NOT MODIFY ITS CONTENT"
    bars = "=" * len(text)
    return f"..\n  {bars}\n  {text}\n  {bars}\n\n"


def build_doxygen_page(name, items):
    content = rst_comment()
    content += name + "\n"
    content += "=" * len(name) + "\n"

    for item in items:
        if isinstance(item, str):
            symbols = item
            title = item
            directive = "function"
        elif len(item) == 2:
            title, symbols = item
            directive = "function"
        else:
            title, symbols, directive = item

        if isinstance(symbols, str):
            symbols = [symbols]

        content += title + "\n"
        content += "-" * len(title) + "\n"

        for symbol in symbols:
            content += f".. doxygen{directive}:: kernel_float::{symbol}\n\n"

    stripped_name = name.lower().replace(" ", "_")
    filename = f"api/{stripped_name}.rst"
    print(f"writing to {filename}")

    with open(filename, "w") as f:
        f.write(content)

    return filename


def build_index_page(groups):
    body = ""
    children = []
    toc = []

    for groupname, symbols in groups.items():
        body += f".. raw:: html\n\n   <h2>{groupname}</h2>\n\n"
        filename = build_doxygen_page(groupname, symbols)
        toc.append(filename)


    title = "API Reference"
    content = rst_comment()
    content += title + "\n" + "=" * len(title) + "\n"
    content += ".. toctree::\n"

    for item in toc:
        content += "   " + item + "\n"

    #filename = "api.rst"
    #print(f"writing to {filename}")
    #with open(filename, "w") as f:
    #    f.write(content)

    return filename

aliases = []
for ty in ["vec", "float", "double", "half", "bfloat16x", ""]:
    if ty != "vec":
        aliases.append(f"{ty}X")

    for i in range(2, 8 + 1):
        aliases.append(f"{ty}{i}")

groups = {
        "Types": [
            ("vector", "vector", "struct"),
            ("Aliases", [
                "unaligned_vec",
                "vec",
            ] + aliases,
            "typedef"),
        ],
        "Primitives": [
            ("range", "range()"),
            ("range", "range(F)"),
            "map",
            "reduce",
            "zip",
            "cast",
            "broadcast",
            "for_each",
        ],
        "Unary Operators": [
            "fill",
            "zeros",
            "ones",
            "negate",
            "bit_not",
            "logical_not",
        ],
        "Binary Operators": [
            "add",
            "subtract",
            "divide",
            "multiply",
            "equal_to",
            "not_equal_to",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "bit_and",
            "bit_or",
            "bit_xor",

            "copysign",
            "hypot",
            ("max", "max(L&&, R&&)"),
            ("min", "min(L&&, R&&)"),
            "nextafter",
            "modf",
            "pow",
            "remainder",
            #"rhypot",
        ],
        "Reductions": [
            "sum",
            ("max", "max(V&&)"),
            ("min", "min(V&&)"),
            "product",
            "all",
            "any",
            "count",
        ],
        "Mathematical": [
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cbrt",
            "ceil",
            "cos",
            "cosh",
            "cospi",
            "erf",
            "erfc",
            "erfcinv",
            "erfcx",
            "erfinv",
            "exp",
            "exp10",
            "exp2",
            "fabs",
            "floor",
            "ilogb",
            "lgamma",
            "log",
            "log10",
            "logb",
            "nearbyint",
            "normcdf",
            "rcbrt",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "tgamma",
            "trunc",
            "rint",
            "rsqrt",
            "round",
            "signbit",
            "isinf",
            "isnan",
        ],
}

build_index_page(groups)
