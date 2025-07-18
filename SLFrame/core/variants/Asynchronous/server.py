import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.extend("../../../")

from ...log.Log import Log

class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = args["server_model"].to(self.device)
        self.args = args
        self.comm = args["comm"]
        # self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]
        self.loss_thred = args["thred"]

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.active_node = 1
        self.train_mode()
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                             lr=args["lr"],
        #                             betas=(0.9, 0.999),
        #                             eps=1e-08,
        #                             weight_decay=0,
        #                             amsgrad=False)
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.total_loss = 0
        self.last_update_loss = 0
        self.state = "A"
        self.delta_loss = 0
        self.client_model = None
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    # def forward_pass(self, acts, labels):
    #     self.acts = acts.to(self.device)
    #     self.labels = labels.to(self.device)
    #     # self.acts = acts
    #     self.optimizer.zero_grad()
    #     self.acts.retain_grad()
    #     logits = self.model(acts)
    #     _, predictions = logits.max(1)
    #     self.loss = self.criterion(logits, labels)
    #     self.total += labels.size(0)
    #     self.correct += predictions.eq(labels).sum().item()

    #     self.total_loss += self.loss
    #     if self.step % self.log_step == 0 and self.phase == "train":
    #         acc = self.correct / self.total
    #         self.log.info("phase={} acc={} loss={} epoch={} and step={}"
    #                       .format("train", acc, self.loss.item(), self.epoch, self.step))

    #         # 用log记录一下准确率之类的信息
    #     if self.phase == "validation":
    #         # self.log.info("phase={} acc={} loss={} epoch={} and step={}"
    #         #               .format("train", acc, self.loss.item(), self.epoch, self.step))
    #         self.val_loss += self.loss.item()
    #         # torch.save(self.model, self.args["model_save_path"].format("server", self.epoch, ""))
    #     self.step += 1
    def forward_pass(self, acts, labels):
        self.acts = acts.to(self.device)
        self.labels = labels.to(self.device)

        if self.phase == "train":
            self.optimizer.zero_grad()
            self.acts.retain_grad()
            logits = self.model(self.acts)
            _, predictions = logits.max(1)
            self.loss = self.criterion(logits, self.labels)
            self.total += labels.size(0)
            self.correct += predictions.eq(self.labels).sum().item()

            self.total_loss += self.loss.item()

            if self.step % self.log_step == 0:
                acc = self.correct / self.total
                self.log.info(f"phase=train acc={acc:.4f} loss={self.loss.item():.4f} epoch={self.epoch} step={self.step}")

            self.step += 1

        else:  # validation phase
            with torch.no_grad():
                logits = self.model(self.acts)
                _, predictions = logits.max(1)
                self.loss = self.criterion(logits, self.labels)
                self.total += labels.size(0)
                self.correct += predictions.eq(self.labels).sum().item()

                self.val_loss += self.loss.item()
                self.step += 1

    def backward_pass(self):
        # self.loss.backward(retain_graph=True)
        self.loss.backward() 
        self.optimizer.step()
        torch.cuda.empty_cache()
        return self.acts.grad

    def validation_over(self):
        # not precise estimation of validation loss
        self.val_loss /= self.step
        acc = self.correct / self.total

        # 这里也要用log记录一下准确率之类的信息
        self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format(self.phase, acc, self.val_loss, self.epoch, self.step))
        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()

    def update_state(self):
        pass
