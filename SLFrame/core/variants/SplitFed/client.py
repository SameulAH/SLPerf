import torch.optim as optim
import logging
import torch
import logging

class SplitNNClient():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 1 if self.rank == self.MAX_RANK else self.rank + 1
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args["lr"],
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0,
                                    amsgrad=False)
        self.local_sample_number = len(self.trainloader)

        self.device = args["device"]

    def forward_pass(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        logging.info("{} forward_pass".format(self.rank))
        return self.acts, labels

    def backward_pass(self, grads):
        self.acts.backward(grads)

        self.optimizer.step()

    """
    如果模型有dropout或者BN层的时候，需要改变模型的模式
    """

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.model.eval()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.model.train()
