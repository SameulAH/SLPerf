import torch
from .message_define import MyMessage
from ...communication.msg_manager import MessageManager
from ...communication.message import Message
from ...log.Log import Log

class AsyncSplitServerManager(MessageManager):
    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.log = Log(self.__class__.__name__, args)
        self.MAX_RANK = args["max_rank"]

    def run(self):
        self.trainer.train_mode()
        num_batches = len(next(iter(self.trainer.args["trainloader"])))
        for batch_idx in range(num_batches):
            acts_labels = []
            for client_rank in range(1, self.MAX_RANK + 1):
                msg = self.receive_message(AsyncSplitMessage.MSG_TYPE_C2S_SEND_ACTS, source=client_rank)
                acts = msg.get(AsyncSplitMessage.MSG_ARG_KEY_ACTS)
                labels = msg.get(AsyncSplitMessage.MSG_ARG_KEY_LABELS)
                acts_labels.append((acts, labels, client_rank))
            for acts, labels, client_rank in acts_labels:
                grads = self.trainer.forward_backward(acts, labels)
                self.send_grads_to_client(client_rank, grads)
        # Optionally: call self.trainer.evaluate() for validation

    def send_grads_to_client(self, receive_id, grads):
        message = Message(AsyncSplitMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(AsyncSplitMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)










# ## server_manager.py
# import logging
# from .message_define import MyMessage
# from ...communication.msg_manager import MessageManager
# from ...communication.message import Message

# class ServerManager(MessageManager):
#     """Handles messages for Parallel Split Learning on the server side."""

#     def __init__(self, args, trainer, backend="MPI"):
#         super().__init__(args, "server", args["comm"], args["rank"], args["max_rank"] + 1, backend)
#         self.trainer = trainer

#     def handle_message_acts_batch(self, msg_params):
#         """Handle forward messages from all clients in a batch."""
#         messages = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)

#         grads = self.trainer.forward_pass_all(messages)

#         backgrad_payload = [{"client_id": cid, "grad": grad} for (cid, grad) in grads]

#         self.send_grads_to_all(backgrad_payload)

#     def send_grads_to_all(self, backgrad_payload):
#         """Send back all gradients to respective clients in a single batch."""
#         message = Message(MyMessage.MSG_TYPE_S2C_GRADS_BATCH, self.rank, None)
#         message.add_params(MyMessage.MSG_ARG_KEY_GRADS, backgrad_payload)
#         self.send_message(message)

#     def register_message_receive_handlers(self):
#         self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS_BATCH,
#                                              self.handle_message_acts_batch)
