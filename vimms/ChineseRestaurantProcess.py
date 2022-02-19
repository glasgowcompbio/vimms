import numpy as np


def discrete_draw(p):
    # samples a discrete number based on a vector of probabilities
    probs = [float(z) / sum(p) for z in p]
    # rv = np.random.multinomial(1, probs)
    return int(np.where(np.random.multinomial(1, probs) == 1)[0])


def Restricted_Crp(alpha, previous_counts, previous_ms2, len_current_ms2):
    # Draws a value from a Chinese Restaurant process, but excludes
    # values already part of the current sample
    n = len(previous_ms2)
    if previous_ms2 == []:
        return 0, [1]
    assign_probs = [None] * (len(previous_counts) + 1)
    index_to_zero = previous_ms2[-(len_current_ms2):]
    for i in range(len(previous_counts)):
        if i in index_to_zero:
            assign_probs[i] = 0
        else:
            assign_probs[i] = previous_counts[i] / (n - 1 + alpha)
    assign_probs[-1] = alpha / (n - 1 + alpha)
    next_crp = discrete_draw(assign_probs)
    if next_crp == (len(previous_counts)):
        previous_counts.append(1)
    else:
        previous_counts[next_crp] += 1
    return next_crp, previous_counts
