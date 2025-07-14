import torch
import torch.nn as nn
import torch.optim as optim
from ...log.Log import Log
import sys
sys.path.extend("../../../")


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]
        self.epoch = 0
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.device = args["device"]
        self.phase = "train"
        self.reset_local_params()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def forward_backward(self, acts, labels):
        # acts, labels: torch.Tensor (CPU)
        acts = acts.to(self.device).requires_grad_()
        labels = labels.to(self.device)
        self.model = self.model.to(self.device)
        logits = self.model(acts)
        loss = self.criterion(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        grads = acts.grad.detach().cpu()
        self.optimizer.step()
        # For logging/metrics
        _, predictions = logits.max(1)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        self.val_loss += loss.item()
        self.step += 1
        return grads

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def evaluate(self, acts_loader, labels_loader):
        self.eval_mode()
        correct = 0
        total = 0
        with torch.no_grad():
            for acts, labels in zip(acts_loader, labels_loader):
                acts, labels = acts.to(self.device), labels.to(self.device)
                outputs = self.model(acts)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = correct / total if total > 0 else 0.0
        self.log.info(f"Server evaluation accuracy: {acc}")
        return acc





# # server.py
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from ...log.Log import Log

# class SplitNNServer:

#     def __init__(self, args):
#         self.log = Log(self.__class__.__name__, args)
#         self.args = args
#         self.model = args["server_model"]
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=5e-4)

#     def forward_pass_all(self, messages):
#         """Forward pass for all messages in a batch."""
#         self.model = self.model.to(self.device)
#         act_vals = []
#         labs = []
#         client_ids = []

#         for (acts, labels, clientId) in messages:
#             act_vals.append(acts.to(self.device))
#             labs.append(labels.to(self.device))
#             client_ids.append(clientId)

#         preds = self.model(torch.stack(act_vals))
#         losses = self.criterion(preds, torch.stack(labs))
#         losses.backward()

#         grads = [acts.grad for acts in act_vals]

#         self.optimizer.step()
#         self.optimizer.zero_grad()

#         return list(zip(client_ids, grads))

