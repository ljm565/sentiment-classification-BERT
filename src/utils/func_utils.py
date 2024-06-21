def labeling(score):
    for i, s in enumerate(score):
        if s < 3:
            score[i] = 0
        elif s == 3:
            score[i] = 1
        else:
            score[i] = 2
    return score


def label_mapping(score):
    if score == 0:
        return "negative"
    elif score == 1:
        return "mediocre"
    return "positive"