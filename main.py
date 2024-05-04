import argparse
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader.dataloader import DataLoader
from models.GCN import GCN
from models.GT import GraphTransformerNet
from data.load_data import load_ZINC
from torch_geometric.datasets import ZINC
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--config', help="Config file")
    parser.add_argument('--model', type=str, help='Nombre del modelo')
    parser.add_argument('--dataset', type=str, help='Dataset Name')
    parser.add_argument('--subset', type=str, help='Particion a utilizar')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = ZINC(root=f'data/{DATASET_NAME}')
    print(dataset[0])
    print(dataset[0].size)
    input("AAA")
    params = config['params']
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    
    if MODEL_NAME == 'GCN':
        model = GCN(dataset.num_features, 32, 1)
    elif MODEL_NAME == 'GT':
        model = GraphTransformerNet(config['net_params'])

    
    dataloader = DataLoader(dataset, batch_size=params['batch_size'])
    print(dataloader)
    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    
    model.to(device)
    for epoch in range(params['epochs']):
        # Forward pass
        for data in tqdm(dataloader):
            data.to(device)
            #data.x = data.x.float()
            # Forward pass
            output = model(data)
            loss = criterion(output, data.y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss
        print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')

    # Evaluate the model
    with torch.no_grad():
        output = model(data.x.float(), data.edge_index)
        mae = torch.mean(torch.abs(output - data.y))
        print(f'MAE: {mae.item():.4f}')
