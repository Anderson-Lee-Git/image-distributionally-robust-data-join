import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def train_one_epoch(X_P, y_P, model, optimizer, batch_size, device):
    correct = 0
    # loop = tqdm(total=len(X_P) // batch_size + 1, leave=True, position=0)
    loss_fn = CrossEntropyLoss()
    for i in range(0, len(X_P), batch_size):
        if i + batch_size <= len(X_P):
            X_batch = X_P[i:i+batch_size]
            y_batch = y_P[i:i+batch_size]
        else:
            X_batch = X_P[i:]
            y_batch = y_P[i:]
        x = torch.tensor(X_batch, dtype=torch.float32).to(device)
        labels = torch.LongTensor(y_batch).to(device)
        output = model(x)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loop.set_description(f"loss={loss.item()}")
        # loop.update(1)
        correct += torch.count_nonzero((torch.argmax(output, dim=1) == labels))
    acc = correct / len(X_P)
    return acc

def evaluate(X_test, y_test, model, batch_size, device):
    correct = 0
    # loop = tqdm(total=len(X_test) // batch_size + 1, leave=True, position=0)
    for i in range(0, len(X_test), batch_size):
        if i + batch_size <= len(X_test):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
        else:
            X_batch = X_test[i:]
            y_batch = y_test[i:]
        x = torch.tensor(X_batch, dtype=torch.float32).to(device)
        labels = torch.LongTensor(y_batch).to(device)
        output = model(x)
        correct += torch.count_nonzero((torch.argmax(output, dim=1) == labels))
        # loop.update(1)
    acc = correct / len(X_test)
    return acc


        
