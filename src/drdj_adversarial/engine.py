import torch
from models import DRDJAdversarial
from tqdm import tqdm

def train_one_epoch(model: DRDJAdversarial,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    args=None):
    model.train()
    loss = 0
    batch_cnt = 0
    acc = 0
    for step, (x1, x2, labels) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output, batch_loss = model.forward_adv(x=x1,
                                               x_other=x2,
                                               labels=labels,
                                               attack_prob=args.attack_prob)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        # calculate acc
        pred = torch.argmax(output, dim=1)
        batch_acc = torch.sum(pred == labels) / len(labels)
        acc += batch_acc.item()
        batch_cnt += 1
    # print(f"pred = {pred}")
    # print(f"labels = {labels}")
    return loss / batch_cnt, acc / batch_cnt

def evaluate(model: DRDJAdversarial, data_loader: torch.utils.data.DataLoader,
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
            output = model.forward_eval(images)
            batch_loss = criterion(output, labels)
            loss += batch_loss.item()
            # calculate acc
            pred = torch.argmax(output, dim=1)
            batch_acc = torch.sum(pred == labels) / len(labels)
            acc += batch_acc.item()
            batch_cnt += 1
    return loss / batch_cnt, acc / batch_cnt