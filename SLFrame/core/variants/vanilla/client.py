import torch.optim as optim
import logging
from ...log.Log import Log
import torch

class SplitNNClient():

    def __init__(self, args):
        self.args=args
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.rank = args["rank"]
        self.log = Log(self.__class__.__name__, args)
        self.MAX_RANK = args["max_rank"]
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 1 if self.rank == self.MAX_RANK else self.rank + 1
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                             lr=args["lr"],
        #                             betas=(0.9, 0.999),
        #                             eps=1e-08,
        #                             weight_decay=0,
        #                             amsgrad=False)
        self.step=0
        self.device = args["device"]

    def forward_pass(self):
        self.log.info(f"Client {self.rank} starting forward pass.")
        #self.model = self.model.to(device)
        self.model = self.model.to(self.device)



        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        self.log.info("{} forward_pass".format(self.rank))
        
        if self.args["save_acts_step"] >0 and self.step<=1 and self.phase=="validation" and self.epoch_count % self.args["save_acts_step"] == 0:
            a=self.acts
            a=a.cpu().detach()
            a=a.numpy()
            f=open("./model_save/acts/Vanilla/A_{}_E_{}_C_{}.txt".format(self.args["partition_alpha"],self.epoch_count,self.rank),"w")

        
            for i in range(a.shape[0]):
                f.write("{} {} {}\n".format(self.rank,labels[i],str(list(a[i].flatten()))))
        
        self.step += 1
        return self.acts, labels

    def backward_pass(self, grads):
        self.log.info(f"Client {self.rank} starting backward pass.")
        self.acts.backward(grads)

        self.optimizer.step()
        self.log.info(f"Client {self.rank} backward pass completed.")

    """
    如果模型有dropout或者BN层的时候，需要改变模型的模式
    """

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.model.eval()
        self.reset_local_params()
        self.phase = "validation"
        self.log.info(f"Client {self.rank} switched to evaluation mode.")

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.model.train()
        self.reset_local_params()
        self.phase = "train"
        self.log.info(f"Client {self.rank} switched to training mode.")

    def print_com_size(self, com_manager):
        self.log.info("worker_num={} epoch_send={} epoch_receive={} total_send={} total_receive={}"
                      .format(self.rank, com_manager.send_thread.tmp_send_size,
                              com_manager.receive_thread.tmp_receive_size,
                              com_manager.send_thread.total_send_size, com_manager.receive_thread.total_receive_size))

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
        
    def local_validate(self):
        self.log.info(f"Client {self.rank} running local validation on client-side model.")
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(self.testloader.dataset, batch_size=self.args["batch_size"], shuffle=False)
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)  # smashed features
                # Optional: Use a linear classifier head for local proxy evaluation
                if self.args.get("local_val_head"):
                    logits = self.args["local_val_head"](outputs)
                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

        if total > 0:
            acc = 100.0 * correct / total
            self.log.info(f"[Local Client {self.rank}] Accuracy on local validation: {acc:.2f}%")
            return acc
        else:
            return None