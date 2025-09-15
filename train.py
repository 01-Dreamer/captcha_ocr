import torch
from torch import nn
import torch.utils.data as Data
from captchaDataset import captchaDataset
from model import captchaNet
import time


if __name__ == '__main__':
    train_data = captchaDataset("./dataset/train")
    train_dataloader = Data.DataLoader(dataset=train_data,
                                  batch_size=128,
                                  num_workers=8,
                                  shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = captchaNet().to(device)

    loss_f = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_time = 0
    epochs = 20

    model.train()
    for epoch in range(epochs):
        loss_total = 0.0
        begin = time.time()
        for _, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, targets)
            loss_total += loss.item() / inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()
        total_time += end - begin
        print("="*10)
        print("Epoch:{}/{}, Loss:{:.4f}".format(epoch+1, epochs, loss_total))
        print("Cost time:{:.0f}m{:.0f}s".format(total_time//60, total_time%60))
        print("="*10)