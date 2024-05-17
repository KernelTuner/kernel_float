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
            if directive == "define":
                content += f".. doxygendefine:: {symbol}\n\n"
            else:
                content += f".. doxygen{directive}:: kernel_float::{symbol}\n\n"

    stripped_name = name.lower().replace(" ", "_").replace("/", "_")
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

aliases = ["scalar", "vec"]
for ty in ["vec"]:
    for i in range(2, 8 + 1):
        aliases.append(f"{ty}{i}")

groups = {
        "Types": [
            ("vector", "vector", "struct"),
            ("Aliases", aliases, "typedef"),
        ],
        "Primitives": [
            "map",
            "reduce",
            "zip",
            "zip_common",
            "make_vec",
            "into_vec",
            "concat",
            "select",
            "for_each",
        ],
        "Conversion": [
            "convert",
            "cast",
            "cast_to",
            "broadcast",
        ],
        "Generation": [
            ("range", "range()"),
            ("range", "range(F fun)"),
            "range_like",
            "each_index",
            "fill",
            "fill_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
        ],
        "Shuffling": [
        #    "concat",
        #    "swizzle",
        #    "first",
        #    "last",
        #    "reversed",
        #    "rotate_left",
        #    "rotate_right",
        ],
        "Unary Operators": [
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
            ("pow", "pow(L&&, R&&)"),
            "remainder",
            #"rhypot",
        ],
        "Reductions": [
            "sum",
            ("max", "max(const V&)"),
            ("min", "min(const V&)"),
            "product",
            "all",
            "any",
            "count",
        ],
        "Mathematical": [
            ("abs", "abs(const V&)"),
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
            ("exp", "exp(const V&)"),
            "exp10",
            "exp2",
            "fabs",
            "floor",
            "ilogb",
            "lgamma",
            ("log", "log(const V&)"),
            "log10",
            "logb",
            "nearbyint",
            "normcdf",
            "rcbrt",
            "sin",
            "sinh",
            ("sqrt", "sqrt(const V&)"),
            "rsqrt",
            "tan",
            "tanh",
            "tgamma",
            "rcp",
            "rint",
            "round",
            "trunc",
            "signbit",
            "isinf",
            "isnan",
        ],
        "Fast math": [
                "fast_exp",
                "fast_log",
                "fast_cos",
                "fast_sin",
                "fast_tan",
                "fast_rcp",
                "fast_sqrt",
                "fast_rsqrt",
                "fast_div",
        ],
        "Conditional": [
            ("where", "where(const C&, const L&, const R&)"),
            ("where", "where(const C&, const L&)"),
            ("where", "where(const C&)"),
        ],
        "Memory read/write": [
            ("read", "read(const T*, const I&, const M&)"),
            ("write", "write(T*, const I&, const V&, const M&)"),

            ("read", "read(const T*)"),
            ("write", "write(T*, const V&)"),

            ("read_aligned", "read_aligned(const T*)"),
            ("write_aligned", "write_aligned(T*, const V&)"),
            "assert_aligned",

            ("vector_ptr", "vector_ptr", "struct"),
        ],
        "Utilities": [
            ("constant", "constant", "struct"),
            ("tiling", "tiling", "struct"),
            ("KERNEL_FLOAT_TILING_FOR", "KERNEL_FLOAT_TILING_FOR", "define"),
        ]
}

build_index_page(groups)
