import torch
import torch.utils
from tqdm import tqdm
from models import Adversarial

def train_one_epoch(model: Adversarial, 
                    dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    option: str = "target",
                    args=None):
    model.train()
    loss = 0
    batch_cnt = 0
    acc = 0
    for step, sample in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        images = sample["image"].to(device, non_blocking=True)
        labels = sample["label"].to(device, non_blocking=True)
        auxs = sample["aux"].to(device, non_blocking=True)
        auxs = torch.argmax(auxs, dim=1)
        if option == "target":
            output, batch_loss = model.forward_target_loss(images, labels)
        else:
            output, batch_loss = model.forward_attr_loss(images, auxs)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        # calculate acc
        pred = torch.argmax(output, dim=1)
        # print(f"pred = {pred}")
        # print(f"labels = {labels}")
        if option == "target":
            batch_acc = torch.sum(pred == labels) / len(labels)
        else:
            batch_acc = torch.sum(pred == auxs) / len(auxs)
        acc += batch_acc.item()
        batch_cnt += 1
    return loss / batch_cnt, acc / batch_cnt

def evaluate(model: Adversarial, 
             dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.CrossEntropyLoss, 
             device: torch.device,
             args=None):
    model.eval()
    losses = []
    correct_cnt = 0
    sample_cnt = 0
    with torch.no_grad():
        for step, sample in enumerate(tqdm(dataloader)):
            if sample == {}:
                continue
            images = sample["image"].to(device, non_blocking=True)
            labels = sample["label"].to(device, non_blocking=True)
            output = model(images)
            batch_loss = criterion(output, labels)
            losses.append(batch_loss.item())
            # calculate acc
            pred = torch.argmax(output, dim=1)
            correct_cnt += torch.sum(pred == labels).item()
            sample_cnt += len(labels)
    return sum(losses) / len(losses), correct_cnt / sample_cnt