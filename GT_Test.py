import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.datasets import ZINC
from models.GraphTansformerNet import GraphTransformerNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

def train(epoch, loss_func):
    model.train()
    for data in tqdm(train_loader, desc="Training Loader"):
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # print(data.edge_attr.unsqueeze(1).shape)
        out = model(x=data.x.float(), edge_index=data.edge_index, edge_attr=data.edge_attr.float().unsqueeze(1), batch=data.batch)
        loss = loss_func(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        train_mae = mean_absolute_error(data.y.detach().numpy(), out.squeeze().detach().numpy())
    return loss, train_mae


@torch.no_grad()
def test(loader):
    model.eval()
    for data in tqdm(loader, desc="Test Loader"):
        data = data.to(device)
       
        out = model(x=data.x.float(), edge_index=data.edge_index, edge_attr=data.edge_attr.float().unsqueeze(1), batch=data.batch)
        loss = criterion(out.squeeze(), data.y)
        test_mae = mean_absolute_error(data.y.detach().numpy(), out.squeeze().detach().numpy())
    return loss, test_mae

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    dataset_train = ZINC(root=f'data/ZINC', split="train")
    dataset_val = ZINC(root=f'data/ZINC', split="val")
    dataset_test = ZINC(root=f'data/ZINC', split="test")

    print(dataset_train[0].num_edge_features)

    epochs = 2
    batch_size = 32
    print(dataset_train.num_edge_features)
    model = GraphTransformerNet(node_dim=dataset_train.num_features,
                                edge_dim=dataset_train.num_edge_features,
                                hidden_dim=128,
                                num_layers=4,
                                num_heads=8)
    print(model)
    total_params = count_parameters(model)
    print("Total de parámetros en el modelo:", total_params)
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)
    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                              min_lr=0.00001)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(epochs):
        train_loss, train_mae = train(epoch, criterion)
        val_loss, val_mae = test(val_loader)
        scheduler.step(val_loss)
        print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f} | Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}')

    _, test_mae = test(test_loader)
    print(f'Test MAE: {test_mae:.4f}')
