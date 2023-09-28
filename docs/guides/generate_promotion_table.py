def name(x):
    if x == ("b", 1):
        return "b"

    return f"{x[0]}{x[1]}"

def promote(a, b):
    x = a[0]
    y = b[0]

    if x == y:
        return (x, max(a[1], b[1]))

    if x == "b":
        return b

    if y == "b":
        return a

    if x in ("f", "bf") and y in ("i", "u"):
        return a

    if y in ("f", "bf") and x in ("i", "u"):
        return b

    if x in ("f", "bf") and y in ("f", "bf"):
        if a[1] > b[1]:
            return a
        elif b[1] > a[1]:
            return b
        else:
            return ("f", a[1] * 2)

    return None


if __name__ == "__main__":
    types = [("b", 1)]
    types += [("i", n) for n in [8, 16, 32, 64]]
    types += [("u", n) for n in [8, 16, 32, 64]]
    types += [("f", n) for n in [8, 16, 32, 64]]

    types.insert(types.index(("f", 32)), ("bf", 16))

    lines = []

    header = [""]
    for a in types:
        header.append(f"**{name(a)}**")
    lines.append(",".join(header))

    for a in types:
        line = [f"**{name(a)}**"]

        for b in types:
            c = promote(a, b)
            line.append(name(c) if c else "x")

        lines.append(",".join(line))

    with open("promotion_table.csv", "w") as f:
        f.write("\n".join(lines))
