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
from core.model.models import german_LR_client, german_LR_server, LeNetClientNetwork1,  LeNetServerNetwork1,\
    adult_LR_client, adult_LR_server, LeNetComplete


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # logging = Logging("init_training_device")
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device(
        "cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == '__main__':
    args = parseFactory(fileType=YAML).factory()
    client_model = LeNetClientNetwork1()
    
    server_model = LeNetServerNetwork1()
    args.load('./config.yaml')

    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # Set the current process_id.

    args["client_model"] = client_model
    args["server_model"] = server_model
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = device

    dataset = datasetFactory(args).factory()  # loader data and partition method

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id) 

    SplitNN_distributed(process_id, args)