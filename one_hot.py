import common
import torch
import torch.nn.functional as F


def text2vec(text):
    vector=torch.zeros((common.captcha_length, common.captcha_char.__len__()))
    for i in range(len(text)):
        vector[i, common.captcha_char.index(text[i])] = 1
    return vector

def vec2text(vector):
    vector=torch.argmax(vector, dim=1)
    text = ""
    for i in vector:
        text += common.captcha_char[i]
    return text

if __name__ == '__main__':
    vec = text2vec("abcd")
    text = vec2text(vec)
    print(vec)
    print(text)
