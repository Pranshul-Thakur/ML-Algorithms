
def hardsigmoid(x):
    return max(0.0, min(1.0, (0.2 * x + 0.5)))

array = [-10, -1, 0, 1, 10]
result = [hardsigmoid(x) for x in array]

print(result)
