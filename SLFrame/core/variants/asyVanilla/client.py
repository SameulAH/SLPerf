import torch.optim as optim
import logging
from ...log.Log import Log

class SplitNNClient():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.SERVER_RANK = args["server_rank"]

        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.device = args["device"]
        self.phase = "train"
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = 3
        self.MAX_EPOCH_PER_NODE = args["epochs"]

        self.log = Log(self.__class__.__name__, args)
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.args = args

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def write_log(self):
        if (self.phase == "train" and self.step%self.log_step==0) or self.phase=="validation":
            self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format(self.phase, self.correct/self.total, self.val_loss, self.epoch_count, self.step))

    def forward_pass(self):
        logging.info("{} begin run_forward_pass".format(self.rank))
        device = self.device
        self.model = self.model.to(device)
        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        logging.info("img size:{}".format(inputs.shape))
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        return self.acts, labels

    def backward_pass(self, grads):
        self.acts.backward(grads)
        self.optimizer.step()

    """
    如果模型有dropout或者BN层的时候，需要改变模型的模式
    """

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.phase = "validation"
        self.model.eval()
        self.reset_local_params()


    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.phase="train"
        self.model.train()
        self.reset_local_params()
