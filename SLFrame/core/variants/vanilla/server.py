from ...log.Log import Log
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


sys.path.extend("../../../")


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]
        self.epoch = 0
        # 经过多少步就记录一次log
        self.log_step = args["log_step"] if args["log_step"] else 50
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        #####
        # Get the log directory path from args or default
        log_dir = "../tensorboad_log/runs"
        os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        
        
        # Initialize per-client metric storage
        #self.total = dict()
        #self.correct = dict()
        #self.val_loss = dict()
        #self.step = dict()





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
        self.log.info(f"Server entered training mode (epoch {self.epoch}).")

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()
        self.log.info(f"Server entered evaluation mode (epoch {self.epoch}).")



    def forward_pass(self, acts, labels):
        
        # if client_id not in self.total:
        #     self.total[client_id] = 0
        #     self.correct[client_id] = 0
        #     self.val_loss[client_id] = 0
        #     self.step[client_id] = 0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the correct device
    
        # Move the model to the correct device (GPU or CPU)
        self.model = self.model.to(device)
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        if self.step % self.log_step == 0 and self.phase == "train":
            acc = self.correct / self.total
            self.log.info("serverforward phase={} acc={} loss={} epoch={} and step={}"
                          .format("train", acc, self.loss.item(), self.epoch, self.step))
            

            # 用log记录一下准确率之类的信息
        if self.phase == "validation":
            #self.log.info("servervalidphase={} acc={} loss={} epoch={} and step={}"
            #              .format("train", acc, self.loss.item(), self.epoch, self.step))
            self.val_loss += self.loss.item()
            # torch.save(self.model, self.args["model_save_path"].format("server", self.epoch, ""))
        self.step += 1

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self.log.info(f"Server completed backward pass at step {self.step}.")
        return self.acts.grad
        

    def validation_over(self):
        # not precise estimation of validation loss
        self.val_loss /= self.step
        acc = self.correct / self.total

        # 这里也要用log记录一下准确率之类的信息
        self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                      .format(self.phase, acc, self.val_loss, self.epoch, self.step))
        # Log to TensorBoard
        self.writer.add_scalar('Validation/Loss', self.val_loss, self.epoch)
        self.writer.add_scalar('Validation/Accuracy', acc, self.epoch)



        # self.log.info(f"Client {client_id} - Validation - acc={acc} loss={val_loss} epoch={self.epoch} step={self.step[client_id]}")

        # self.writer.add_scalar(f'Client/{client_id}/Validation/Loss', val_loss, self.epoch)
        # self.writer.add_scalar(f'Client/{client_id}/Validation/Accuracy', acc, self.epoch)

        # # Reset per-client stats after validation
        # self.val_loss[client_id] = 0
        # self.correct[client_id] = 0
        # self.total[client_id] = 0
        # self.step[client_id] = 0
        
        
        
        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()
        
        
        
        
        
    def training_over(self):
        train_loss = self.val_loss / self.step if self.step > 0 else 0.0
        acc = self.correct / self.total if self.total > 0 else 0.0

        self.log.info("server phase=train acc={} loss={} epoch={} and step={}"
                    .format(acc, train_loss, self.epoch, self.step))
        self.writer.add_scalar('Train/Loss', avg_loss, self.epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, self.epoch)


        # self.log.info(f"Client {client_id} - Train - acc={acc} loss={train_loss} epoch={self.epoch} step={self.step[client_id]}")

        # self.writer.add_scalar(f'Client/{client_id}/Train/Loss', train_loss, self.epoch)
        # self.writer.add_scalar(f'Client/{client_id}/Train/Accuracy', acc, self.epoch)

        # # Reset per-client stats after training
        # self.val_loss[client_id] = 0
        # self.correct[client_id] = 0
        # self.total[client_id] = 0
        # self.step[client_id] = 0
        
        
        
        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1

    # def reset_local_params(self):
    #     self.total = 0
    #     self.correct = 0
    #     self.val_loss = 0
    #     self.step = 0
    #     self.batch_idx = 0
