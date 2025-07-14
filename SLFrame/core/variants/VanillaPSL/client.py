import torch
import torch.optim as optim
from ...log.Log import Log

class SplitNNClient():
    def __init__(self, args):
        self.args = args
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.rank = args["rank"]
        self.log = Log(self.__class__.__name__, args)
        self.MAX_RANK = args["max_rank"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        self.device = args["device"]
        self.step = 0
        self.epoch_count = 0
        self.phase = "train"
        self.reset_local_params()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def forward_pass(self, inputs, labels):
        self.model = self.model.to(self.device)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        acts = self.model(inputs)
        self.acts = acts
        self.labels = labels
        return acts.detach().cpu(), labels.detach().cpu()

    def backward_pass(self, grads):
        # grads: torch.Tensor, same shape as self.acts
        self.acts.backward(grads.to(self.device))
        self.optimizer.step()

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def evaluate(self):
        self.eval_mode()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = correct / total if total > 0 else 0.0
        self.log.info(f"Client {self.rank} evaluation accuracy: {acc}")
        return acc























# #Client.py
# import torch.optim as optim
# import logging
# from ...log.Log import Log

# class SplitNNClient:

#     def __init__(self, args):
#         self.args = args
#         self.comm = args["comm"]
#         self.model = args["client_model"]
#         self.trainloader = args["trainloader"]
#         self.testloader = args["testloader"]
#         self.rank = args["rank"]
#         self.log = Log(self.__class__.__name__, args)
#         self.device = args["device"]
#         self.SERVER_RANK = args["server_rank"]
#         self.optimizer = optim.SGD(self.model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=5e-4)

#         self.acts = None

#     def forward_pass(self, inputs, labels):
#         print("Perform forward pass and return the smashed data and labels immediately.")
#         self.model = self.model.to(self.device)
#         inputs, labels = inputs.to(self.device), labels.to(self.device)

#         self.optimizer.zero_grad()
#         self.acts = self.model(inputs)  # Keep graph for backpropagation

#         return self.acts, labels  # <- Do NOT detach here
    
#     def train_mode(self):
#         self.dataloader = iter(self.trainloader)
#         self.model.train()
#         self.reset_local_params()
#         self.phase = "train"  


#     def backward_pass(self, grads):
#         print("Perform backward pass once server's gradients are received.")
#         self.acts.requires_grad = True
#         self.acts.backward(grads.to(self.device))
#         self.optimizer.step()

    
#     def eval_mode(self):
#         self.dataloader = iter(self.testloader)
#         self.model.eval()
#         self.reset_local_params()
#         self.phase = "validation"
#         self.log.info(f"Client {self.rank} switched to evaluation mode.")
    

