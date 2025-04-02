import numpy as np

def bayes_theorem(a, b, la, lb):
    total = (a * la) + (b * lb)
    post_prob_a = (a * la) / total
    return post_prob_a

a = 0.5
b = 0.5

la = 4/7
lb = 6/10

probability = bayes_theorem(a, b, la, lb)
print(round(probability, 3))