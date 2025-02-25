import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Remove GNN imports and add CNN components
from core.model.cnn import cifar10_client, cifar10_server  # Use existing CNN components
from core.splitApi import SplitNN_distributed, SplitNN_init
from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, YAML
### Added  ##################################
from Parse.parseFactory import parseFactory
import os

import yaml

# Assume parseFactory(fileType=YAML).factory() loads a dictionary (args) from a YAML file
# This is a simulation of that function loading the YAML (adjust this if you are using a different method):
with open('config.yaml', 'r') as file:  # Load the YAML file
    args = yaml.safe_load(file)  # Load as a dictionary

# Access the save_dir from the loaded dictionary
# Since save_dirs is a list, you probably want the first directory, so access it with args['save_dirs'][0]
save_dir = args.get("save_dirs", ["./model_save/default"])[0]  # Default to the first directory if not in config

# Now ensure the directory exists, and create it if it doesn't
os.makedirs(save_dir, exist_ok=True)

##################################
# Simple CNN Model Definition
class CNNSplitClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # CIFAR-10 has 3 channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class CNNSplitServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for CIFAR-10
        
    def forward(self, x):
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#added by ismail
torch.serialization.add_safe_globals(['CNNSplitClient'])
#### 
# Initialize training device
def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

if __name__ == '__main__':
    # Initialize framework
    args = parseFactory(fileType=YAML).factory()
    args.load('./config.yaml')
    
    # Initialize communication
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id

    # Create CNN models
    client_model = CNNSplitClient()
    server_model = CNNSplitServer()
    
    # Assign models to args
    args["client_model"] = client_model
    args["server_model"] = server_model

    # Set device (force CPU for compatibility)
    args["device"] = "cpu"

    # Load dataset
    dataset = datasetFactory(args).factory()
    
    # Get partitioned data
    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)

    # Initialize logging
    log = Log("main", args)

    # Start distributed training
    SplitNN_distributed(process_id, args)