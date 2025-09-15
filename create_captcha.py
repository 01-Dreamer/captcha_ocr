from captcha.image import ImageCaptcha
import common
import random
import time
import os

captcha_char = common.captcha_char
captcha_length = common.captcha_length

train_size = common.train_size
test_size = common.test_size

if __name__ == '__main__':
    os.makedirs("./dataset/train", exist_ok=True)
    os.makedirs("./dataset/test", exist_ok=True)

    image =ImageCaptcha()
    for i in range(train_size):
        image_val = "".join(random.sample(captcha_char, captcha_length))
        image_name = "./dataset/train/{}_{}.png".format(image_val, int(time.time()))
        image.write(image_val, image_name)

    for i in range(test_size):
        image_val = "".join(random.sample(captcha_char, captcha_length))
        image_name = "./dataset/test/{}_{}.png".format(image_val, int(time.time()))
        image.write(image_val, image_name)
        