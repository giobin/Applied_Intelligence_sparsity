import pandas

class result:
    def __init__(self, bleu, sparsity):
        self.sp = sparsity
        self.bleu = bleu

    def __str__(self):
        return f'{self.sp},{self.bleu}'


def takeSecond(elem):
    res = elem.sp
    return res

def takeBleu(elem):
    res = elem.bleu
    return res

if __name__ == '__main__':
    df = pandas.read_csv('transformer_variational_dropout.csv')
    elements = []
    for index, elem in df.iterrows():
        elem = elem.tolist()
        elements.append(result(elem[6], elem[7]))

    elements.sort(key=takeSecond)
    print(len(elements))

    i = 0
    maximum = []
    temp = []
    for elem in elements:
        if i % 2 != 0:
            temp.append(elem)
        else:
            if len(temp) > 0:
                temp.sort(key=takeBleu, reverse=True)
                maximum.append(temp[0])
                temp.clear()
        i += 1

    for elem in maximum:
        print(elem)