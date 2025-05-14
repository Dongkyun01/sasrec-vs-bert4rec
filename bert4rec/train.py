import torch

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)