import logging


class AsyncController(object):
    def __init__(self, args):
        self.args = args
        async_sec_per_step = args.async_sec_per_step

    def get_next_task(self, global_virtual_clock):
        pass

    def list_tasks(self, global_virtual_clock):
        pass

    def select_participant(self, available_clients):
        pass

    def update_future(self, sampled_clients, global_virtual_clock):
        pass