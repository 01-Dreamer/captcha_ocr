import torch
import torch.utils.data as Data
from captchaDataset import captchaDataset
from torchvision  import transforms
from PIL import Image
import common


if __name__ == '__main__':
    ############################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("best.pth", map_location=device)
    transform = transforms.Compose([
    transforms.Resize((60, 160)),
    transforms.Grayscale(),
    transforms.ToTensor()
    ])

	model.eval()
    img = Image.open("./dataset/test/0icv_1757934185.png")
    input = transform(img).unsqueeze(0).to(device)
    output = model(input)
    output = output.view(-1, common.captcha_length, common.captcha_char.__len__())
    output = torch.argmax(output, dim=2)[0]
    
    for i in range(common.captcha_length):
        print(common.captcha_char[output[i]], end='')
    print()


    exit(0)
    ############################################################
    test_data = captchaDataset("./dataset/test")
    test_dataloader = Data.DataLoader(dataset=test_data,
                                  batch_size=32,
                                  num_workers=2,
                                  shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("best.pth", map_location=device)

    test_toal = 0
    test_correct = 0

    model.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.view(-1, common.captcha_length, common.captcha_char.__len__())
            outputs = torch.argmax(outputs, dim=2)
            targets = targets.view(-1, common.captcha_length, common.captcha_char.__len__())
            targets = torch.argmax(targets, dim=2)

            test_correct += torch.sum(torch.all(outputs == targets, dim=1)).item()
            test_toal += targets.size(0)
    
    print("Test Accuracy: {:.2f}%".format(test_correct / test_toal * 100))

