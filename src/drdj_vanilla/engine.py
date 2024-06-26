import torch
from models import DRDJVanilla
from tqdm import tqdm

def train_one_epoch(model: DRDJVanilla,
                    data_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    include_max_term=True,
                    include_norm=True,
                    args=None):
    model.train()
    loss = 0
    batch_cnt = 0
    acc = 0
    for step, sample in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        x1 = sample["image_1"].to(device, non_blocking=True)
        x2 = sample["image_2"].to(device, non_blocking=True)
        aux = sample["aux"].to(device, non_blocking=True)
        if "dist_weight" in sample:
            dist_weight = sample["dist_weight"].to(device, non_blocking=True)
        else:
            dist_weight = None
        labels = sample["label"].to(device, non_blocking=True)
        output, batch_loss = model.forward_loss(x=x1,
                                                x_other=x2,
                                                aux=aux,
                                                labels=labels,
                                                dist_weight=dist_weight,
                                                include_max_term=include_max_term,
                                                include_norm=include_norm)
        batch_loss.backward()
        optimizer.step()
        # enforce alpha_a and alpha_p >= 0
        # model.alpha_a.clamp_min(min=0.0)
        # model.alpha_p.clamp_min(min=0.0)
        loss += batch_loss.item()
        # calculate acc
        pred = torch.argmax(output, dim=1)
        batch_acc = torch.sum(pred == labels) / len(labels)
        acc += batch_acc.item()
        batch_cnt += 1
    top_aux_weight_per_class = torch.argsort(input=model.fc.weight[:, -20:], dim=1, descending=True)[:, 0]
    classes_per_aux = {}
    for i, aux in enumerate(top_aux_weight_per_class):
        aux = aux.item()
        if aux not in classes_per_aux:
            classes_per_aux[aux] = []
        classes_per_aux[aux].append(i)
    print(classes_per_aux)
    # print(f"pred = {pred}")
    # print(f"labels = {labels}")
    return loss / batch_cnt, acc / batch_cnt

def evaluate(model: DRDJVanilla, data_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.CrossEntropyLoss, device: torch.device,
             args=None):
    model.eval()
    loss = 0
    acc = 0
    batch_cnt = 0
    with torch.no_grad():
        for step, sample in enumerate(tqdm(data_loader)):
            if sample == {}:
                continue
            images = sample["image"].to(device, non_blocking=True)
            labels = sample["label"].to(device, non_blocking=True)
            if "aux" in sample:
                auxs = sample["aux"].to(device, non_blocking=True)
            else:
                auxs = None
            output = model.forward_eval(images, auxs)
            batch_loss = criterion(output, labels)
            loss += batch_loss.item()
            # calculate acc
            pred = torch.argmax(output, dim=1)
            batch_acc = torch.sum(pred == labels) / len(labels)
            acc += batch_acc.item()
            batch_cnt += 1
    return loss / batch_cnt, acc / batch_cnt