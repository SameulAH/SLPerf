import torch
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log
from .message_define import MyMessage

class ClientManager(MessageManager):
    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        self.round_idx = 0

    def run(self):
        self.trainer.train_mode()
        for batch_idx, (inputs, labels) in enumerate(self.trainer.trainloader):
            acts, labels = self.trainer.forward_pass(inputs, labels)
            self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
            grads = self.receive_grads_from_server()
            self.trainer.backward_pass(grads)
        # Optionally: call self.trainer.evaluate() for validation

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        message = Message(AsyncSplitMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(AsyncSplitMessage.MSG_ARG_KEY_ACTS, acts)
        message.add_params(AsyncSplitMessage.MSG_ARG_KEY_LABELS, labels)
        self.send_message(message)

    def receive_grads_from_server(self):
        msg = self.receive_message(AsyncSplitMessage.MSG_TYPE_S2C_GRADS)
        grads = msg.get(AsyncSplitMessage.MSG_ARG_KEY_GRADS)
        return grads
    
    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        message = Message(AsyncSplitMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(AsyncSplitMessage.MSG_ARG_KEY_ACTS, acts)
        message.add_params(AsyncSplitMessage.MSG_ARG_KEY_LABELS, labels)
        self.send_message(message)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(AsyncSplitMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)









#
# # import logging
# # import torch
# # import time
# # from .message_define import MyMessage
# # from ...communication.msg_manager import MessageManager
# # from ...communication.message import Message
# # from ...log.Log import Log

# # class ClientManager(MessageManager):
# #     """
# #     Initializes a pool of clients and performs forward pass in batch.
# #     """

# #     def __init__(self, args, trainer, backend="MPI"):
# #         super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
# #         self.trainer = trainer
# #         self.trainer.train_mode()
# #         self.log = Log(self.__class__.__name__, args)

# #     def run_parallel_forward_pass(self):
# #         """Perform forward pass on all batches in the loader and send in a batch."""
# #         messages = []

# #         for inputs, labels in self.trainer.trainloader:
# #             acts, labs = self.trainer.forward_pass(inputs, labels)
# #             messages.append((acts, labs, self.rank))

# #         self.send_activations_batch_to_server(messages)

# #     def send_activations_batch_to_server(self, messages):
# #         """Send all forward messages in a single batch to server."""
# #         message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS_BATCH, self.rank, self.trainer.SERVER_RANK)
# #         message.add_params(MyMessage.MSG_ARG_KEY_ACTS, messages)
# #         self.send_message(message)

# #     def handle_message_gradients_batch(self, msg_params):
# #         """Receive back batch of gradients from server and apply backward pass."""
# #         grads_payload = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)

# #         for (clientId, grad) in grads_payload:
# #             if clientId == self.rank:
# #                 self.trainer.backward_pass(grad)

# #         self.log.info("Client {} finished backward pass.".format(self.rank))


# ## ClientManager.py
# import logging
# import torch
# import time
# from .message_define import MyMessage
# from ...communication.msg_manager import MessageManager
# from ...communication.message import Message
# from ...log.Log import Log

# class ClientManager(MessageManager):
#     """
#     Initializes a pool of clients and performs forward pass in batch.
#     """

#     def __init__(self, args, trainer, backend="MPI"):
#         super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
#         self.trainer = trainer
#         self.trainer.train_mode()
#         self.log = Log(self.__class__.__name__, args)

#         # Register handlers immediately upon instantiation
#         self.register_message_receive_handlers()

#     def run_parallel_forward_pass(self):
#         """Perform forward pass on all batches in the loader and send in a batch."""
#         messages = []

#         for inputs, labels in self.trainer.trainloader:
#             acts, labs = self.trainer.forward_pass(inputs, labels)
#             messages.append((acts, labs, self.rank))

#         self.send_activations_batch_to_server(messages)

#     def send_activations_batch_to_server(self, messages):
#         """Send all forward messages in a single batch to server."""
#         message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS_BATCH, self.rank, self.trainer.SERVER_RANK)
#         message.add_params(MyMessage.MSG_ARG_KEY_ACTS, messages)
#         self.send_message(message)

#     def handle_message_gradients_batch(self, msg_params):
#         """Receive back batch of gradients from server and apply backward pass."""
#         grads_payload = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)

#         for (clientId, grad) in grads_payload:
#             if clientId == self.rank:
#                 self.trainer.backward_pass(grad)

#         self.log.info("Client {} finished backward pass.".format(self.rank))

#     def register_message_receive_handlers(self):
#         """Register handlers to respond to messages from the server."""
#         self.register_message_receive_handler(
#             MyMessage.MSG_TYPE_S2C_GRADS_BATCH,
#             self.handle_message_gradients_batch
#         )
