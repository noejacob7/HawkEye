"""
Simple broadcast message bus for UAV intercommunication.
"""

class MessageBus:
    def __init__(self):
        self.subscribers = []  # list of callbacks

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def broadcast(self, sender_id, data):
        for cb in self.subscribers:
            cb(sender_id, data)
