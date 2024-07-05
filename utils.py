import math


def clean(values):
    return [v for v in values if not math.isnan(v)]


def mean(values):
    total = 0.0

    for val in values:
        total += val

    return total / len(values)


def std(values):
    total = 0.0
    mean_val = mean(values)

    for v in values:
        total += (v - mean_val) * (v - mean_val)

    return math.sqrt(total / len(values))


def min_val(values):
    res = values[0]

    for i in range(1, len(values)):
        if values[i] < res:
            res = values[i]

    return res


def max_val(values):
    res = values[0]

    for i in range(1, len(values)):
        if values[i] > res:
            res = values[i]

    return res


def sort(values):
    for i in range(len(values)):
        inf = values[i]
        inf_idx = i
        for j in range(i, len(values)):
            if values[j] < inf:
                inf = values[j]
                inf_idx = j
        values[i], values[inf_idx] = values[inf_idx], values[i]


def quantile(values, q):
    pos = q * (len(values) + 1) - 1
    idx = int(pos)
    if idx == pos:
        return values[idx]

    return values[idx] + (values[idx + 1] - values[idx]) * (pos - idx)
