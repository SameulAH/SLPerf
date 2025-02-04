# Adjust imports to reflect your module paths
from core.process.baseClient import BaseClient
from core.process.baseServer import BaseServer
from core.dataset.datasetFactory import datasetFactory
from core.communication.mpi.mpi_com_mananger import MpiCommunicationManager

import torch
from SLFrame import SplitNN_init, datasetFactory, parseFactory, SplitNN_distributed
from SLFrame.models import LeNetClientNetwork, LeNetServerNetwork

if __name__ == '__main__':
    # Load configuration
    args = parseFactory(fileType='YAML').factory()
    args.load('./config.yaml')  # Ensure config.yaml has variants_type: vanilla

    # Initialize models
    client_model = LeNetClientNetwork()
    server_model = LeNetServerNetwork()

    # Set up MPI communication
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id
    args["client_model"] = client_model
    args["server_model"] = server_model

    # Set device (CPU/GPU)
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = device

    # Load dataset
    dataset = datasetFactory(args).factory()
    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)

    # Run vanilla split learning
    SplitNN_distributed(process_id, args)