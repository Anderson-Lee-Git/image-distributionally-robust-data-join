import torch
from tqdm import tqdm

def train_one_epoch(X_P, X_A, y_P, d_X, matchings, 
                    model_P, model_A, optimizer_P, optimizer_A, batch_size, device):
    loop = tqdm(total=len(matchings) // batch_size + 1, leave=True, position=0)
    correct_A = correct_P = 0
    for i in range(0, len(matchings), batch_size):
        if i + batch_size < len(matchings):
            idx = matchings[i:i+batch_size]
        else:
            idx = matchings[i:]
        B, _ = idx.shape
        idx_A = idx[:, 0]
        idx_P = idx[:, 1]
        x = torch.tensor(X_P[idx_P], dtype=torch.float32).to(device)
        x_other = torch.tensor(X_A[idx_A, :d_X], dtype=torch.float32).to(device)
        aux = torch.tensor(X_A[idx_A, d_X:], dtype=torch.float32).to(device)
        labels = torch.LongTensor(y_P[idx_P]).to(device)
        output_P, loss_P = model_P.forward_loss(x=x, x_other=x_other, aux=aux, labels=labels, dist_weight=None, include_max_term=True, include_norm=True)
        optimizer_P.zero_grad()
        loss_P.backward()
        optimizer_P.step()
        output_A, loss_A = model_A.forward_loss(x=x, x_other=x_other, aux=aux, labels=labels, dist_weight=None, include_max_term=True, include_norm=True)
        optimizer_A.zero_grad()
        loss_A.backward()
        optimizer_A.step()
        loop.set_description(f"loss_P={loss_P.item():.3f}, loss_A={loss_A.item():.3f}")
        loop.update(1)
        # update accuracy
        correct_P += torch.count_nonzero(torch.argmax(output_P, dim=1) == labels)
        correct_A += torch.count_nonzero(torch.argmax(output_A, dim=1) == labels)
    acc_P = correct_P / len(matchings)
    acc_A = correct_A / len(matchings)
    return acc_P, acc_A

def evaluate(X_test, y_test, d_X, model, batch_size, device):
    loop = tqdm(total=len(X_test) // batch_size + 1, leave=True, position=0)
    correct = 0
    for i in range(0, len(X_test), batch_size):
        if i + batch_size < len(X_test):
            X_batch = X_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
        else:
            X_batch = X_test[i:]
            y_batch = y_test[i:]
        B, _ = X_batch.shape
        x = torch.tensor(X_batch[:, :d_X], dtype=torch.float32).to(device)
        aux = torch.tensor(X_batch[:, d_X:], dtype=torch.float32).to(device)
        labels = torch.LongTensor(y_batch).to(device)
        output = model.forward_eval(x, aux)
        # update accuracy
        correct += torch.count_nonzero(torch.argmax(output, dim=1) == labels)
        loop.update(1)
    acc = correct / len(X_test)
    return acc