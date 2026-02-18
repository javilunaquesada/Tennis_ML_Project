import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds = (torch.tensor(all_probs) >= threshold).int().numpy()

    acc = accuracy_score(all_targets, preds)
    auc = roc_auc_score(all_targets, all_probs)

    return acc, auc
