#client_manager.py
import logging
import torch
import time
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log
import core.model.models
import torch.serialization
import torch
import torch.nn as nn
import torch.serialization

    # allowlist Sequential globally
torch.serialization.add_safe_globals([nn.Sequential, nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear])
torch.serialization.add_safe_globals([nn.Sequential, nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear])

    ####################

class ClientManager(MessageManager):
    """
    args里面要有MPI的 comm, rank, max_rank(也就是comm.size()-1) 其他的暂时不用
    trainer就是SplitNNClient的一个实例
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        self.round_idx = 0

    def run(self):
        logging.info(f"Client {self.rank} is starting a training session.")
        if self.rank == 1:
            self.log.info("{} begin run_forward_pass".format(self.trainer.rank))
            self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        self.log.info("{} run_forward_pass".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()

        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            # if self.trainer.rank ==
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            if self.trainer.rank != self.trainer.MAX_RANK:
                self.send_test_semaphore_to_client(self.trainer.node_right)
            else:
                self.send_semaphore_to_client(self.trainer.node_right)

        self.trainer.batch_idx = 0

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_TEST_SEMAPHORE,
                                              self.handle_message_test_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MODEL,
                                              self.handle_message_model_param_from_server)

    def handle_message_semaphore(self, msg_params):
        self.log.info("Client {} received test semaphore from server. Starting evaluation...".format(self.rank))
        # no point in checking the semaphore message
        self.log.info("client{} recv sema".format(self.rank))
        self.trainer.train_mode()
        self.trainer.model.load_state_dict(torch.load(self.args["model_tmp_path"]))
        # self.trainer.model = torch.load(self.args["model_tmp_path"])
        self.trainer.model = torch.load(self.args["model_tmp_path"], weights_only=True)
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        self.log.info("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            self.send_model_param_to_fed_server(0)
            if self.trainer.rank != self.trainer.MAX_RANK:
                self.send_semaphore_to_client(self.trainer.node_right)
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.round_idx))
            # torch.save(self.trainer.model.state_dict(), self.args["model_tmp_path"])
            # torch.save(self.trainer.model, self.args["model_tmp_path"])
            # while True:
            #     if self.com_manager.q_receiver.qsize() > 0:
            #         msg_params = self.com_manager.q_receiver.get()
            #         self.log.info(msg_params)
            #
            #         self.com_manager.notify(msg_params)
            #         break
            #     else:
            #         time.sleep(0.5)
            # self.run_eval()
        else:
            self.run_forward_pass()

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_test_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_TEST_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        self.log.info("client {} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        # self.log.info(model_param["block1.0.weight"])
        self.trainer.model.load_state_dict(model_param)

        if self.trainer.rank == 1:
            self.run_eval()

#    def send_model_param_to_fed_server(self, receive_id):
#        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
#        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.model.state_dict())
#        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
#        self.send_message(message)
    def send_model_param_to_fed_server(self, receive_id):
        model_params = self.trainer.model.state_dict()
        self.log.info(f"Client {self.rank} is sending weights to server.")
        for key, val in model_params.items():
            logging.debug(f"Client {self.rank} - {key}: {val.detach().cpu().numpy()}")

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, model_params)
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        self.send_message(message)


    def handle_message_test_semaphore(self, msg_params):
        self.run_eval()
