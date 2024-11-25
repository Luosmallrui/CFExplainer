import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data, Batch
import os
from pathlib import Path
import numpy as np

class ContractGraphDataset(Dataset):
    def __init__(self, root_dir, partition='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
        
        # Load all .pt files
        self.file_list = [file for file in Path(root_dir).glob("*.pt")]
        
        # Partition the dataset
        total_size = len(self.file_list)
        all_indices = list(range(total_size))
        
        if partition == 'train':
            self._indices = all_indices[:int(0.7 * total_size)]
        elif partition == 'val':
            self._indices = all_indices[int(0.7 * total_size):int(0.85 * total_size)]
        else:  # test
            self._indices = all_indices[int(0.85 * total_size):]

    def len(self):
        return len(self._indices)

    def get(self, idx):
        # Get actual file index
        file_idx = self._indices[idx]
        file_path = self.file_list[file_idx]
        
        # Load .pt file
        data = torch.load(file_path)
        
        # Convert all numpy arrays to torch tensors (if necessary)
        if isinstance(data.x, np.ndarray):
            data.x = torch.tensor(data.x)
        if isinstance(data.edge_index, np.ndarray):
            data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
        if hasattr(data, '_VULN') and isinstance(data._VULN, np.ndarray):
            data._VULN = torch.tensor(data._VULN)

        # If transformation is needed, apply here
        if self.transform:
            data = self.transform(data)
            
        return data

    def __len__(self):
        return self.len()

def collate(batch):
    """
    Custom collate function for batching.
    """
    max_nodes = max(data.x.size(0) for data in batch)  # Max number of nodes in batch
    batch_x = []
    batch_edge_index = []
    batch_mask = []
    batch_y = []
    
    cum_nodes = 0
    for i, data in enumerate(batch):
        num_nodes = data.x.size(0)
        
        # Pad node features to max_nodes
        padded_x = torch.zeros((max_nodes, data.x.size(1)), dtype=data.x.dtype)
        padded_x[:num_nodes] = data.x
        batch_x.append(padded_x)
        
        # Adjust edge indices
        if data.edge_index.size(1) > 0:
            edge_index = data.edge_index + cum_nodes
            batch_edge_index.append(edge_index)
        
        # Create mask
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:num_nodes] = True
        batch_mask.append(mask)
        
        # Collect labels (assuming labels are stored in _VULN)
        if hasattr(data, '_VULN'):
            # Ensure that _VULN is a tensor
            vuln_label = data._VULN if isinstance(data._VULN, torch.Tensor) else torch.tensor(data._VULN, dtype=torch.float32)
            batch_y.append(vuln_label)
        
        cum_nodes += num_nodes
    
    # Stack all the batched data
    batch_x = torch.stack(batch_x)
    batch_edge_index = torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0), dtype=torch.long)
    batch_mask = torch.stack(batch_mask)
    
    if batch_y:
        batch_y = torch.stack(batch_y)
    else:
        batch_y = None
    
    return Batch(
        x=batch_x,
        edge_index=batch_edge_index,
        mask=batch_mask,
        y=batch_y,
        batch=torch.arange(len(batch)).repeat_interleave(max_nodes)
    )

def load_datasets(args):
    # Create datasets
    train_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='train'
    )
    
    valid_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='val'
    )
    
    test_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='test'
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate
    )
    
    return train_dataloader, valid_dataloader, test_dataloader

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Set data directory and batch size
    args.data_dir = "/root/autodl-tmp/SCcfg/processed"  # Replace with .pt file directory
    args.batch_size = 128  # Set batch size
    
    # Load data loaders
    train_dataloader, valid_dataloader, test_dataloader = load_datasets(args)
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(valid_dataloader.dataset)}")
    print(f"Testing samples: {len(test_dataloader.dataset)}")

if __name__ == '__main__':
    main()