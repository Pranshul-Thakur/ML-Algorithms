import math

def softmax(array):
    expi = [math.exp(i) for i in array]
    sum_expi = sum(expi)
    # return [i / sum_expi for i in expi] normal method
    return [round(i / sum_expi, 3) for i in expi]

array = [1, 2, 3]

probability = softmax(array)

print(probability)