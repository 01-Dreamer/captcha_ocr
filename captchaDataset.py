import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision  import transforms
import one_hot


class captchaDataset(Dataset):
    def __init__(self,root_dir):
       super(captchaDataset, self).__init__()
       self.list_image_path=[os.path.join(root_dir, image_name) for image_name in os.listdir(root_dir)]
       self.transforms=transforms.Compose([
           transforms.Resize((60, 160)),
           transforms.ToTensor(),
           transforms.Grayscale()
       ])

    def __getitem__(self, index):
        image_path = self.list_image_path[index]
        image_name=image_path.split("/")[-1]
        img = Image.open(image_path)
        img_input = self.transforms(img)
        img_lable = image_name.split("_")[0]
        img_lable = one_hot.text2vec(img_lable)
        img_lable = img_lable.view(1,-1)[0]
        return img_input, img_lable
    
    def __len__(self):
        return self.list_image_path.__len__()

if __name__ == '__main__':
    captchaDataset = captchaDataset("./dataset/train")
    img, label = captchaDataset[0]
    print(img.shape)
    print(label.shape)
