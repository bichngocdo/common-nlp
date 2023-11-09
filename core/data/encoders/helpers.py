def recursive_apply(fn, sequence):
    result = list()
    if not isinstance(sequence, list):
        return fn(sequence)
    else:
        for item in sequence:
            result.append(recursive_apply(fn, item))
    return result


def flatten_list(l):
    if isinstance(l, list):
        for seq in l:
            for item in flatten_list(seq):
                yield item
    else:
        yield l
