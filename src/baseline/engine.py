import torch
from models.resnet import ResNet
from tqdm import tqdm

def train_one_epoch(model: ResNet, data_loader: torch.utils.data.DataLoader,
                    criterion,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    args=None):
    model.train()
    optimizer.zero_grad()
    loss = 0
    batch_cnt = 0
    acc = 0
    for step, (x, labels) in enumerate(tqdm(data_loader)):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(x)
        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        # calculate acc
        pred = torch.argmax(output, dim=1)
        # print(f"pred = {pred}")
        # print(f"labels = {labels}")
        batch_acc = torch.sum(pred == labels) / len(labels)
        acc += batch_acc.item()
        batch_cnt += 1
    return loss / batch_cnt, acc / batch_cnt

def evaluate(model: ResNet, data_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.CrossEntropyLoss, device: torch.device,
             args=None):
    model.eval()
    loss = 0
    acc = 0
    batch_cnt = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            batch_loss = criterion(output, labels)
            loss += batch_loss.item()
            # calculate acc
            pred = torch.argmax(output, dim=1)
            batch_acc = torch.sum(pred == labels) / len(labels)
            acc += batch_acc.item()
            batch_cnt += 1
    return loss / batch_cnt, acc / batch_cnt