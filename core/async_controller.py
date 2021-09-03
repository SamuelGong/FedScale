import logging
from random import Random


class AsyncController(object):
    def __init__(self, args, sample_seed=233):
        self.args = args
        self.mode = args.sync_mode
        self.async_sec_per_step = args.async_sec_per_step
        self.client_end_time_map = {}
        self.client_fake_end_time_map = {}
        # self.end_time_client_dict = {}
        # self.fake_end_time_client_dict = {}
        self.async_num_issues_max = args.async_num_issues_max
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.next_task_list = None
        self.next_task_list_idx = 0

    def get_next_task(self, cur_time):
        if self.next_task_list is None:
            self.next_task_list = []
            for k, v in self.client_end_time_map.items():
                if v <= cur_time:
                    self.next_task_list.append(k)

        if self.next_task_list_idx < len(self.next_task_list):
            clientId = self.next_task_list[self.next_task_list_idx]
            self.next_task_list_idx += 1
            return clientId
        else:
            return None

    def refresh_next_task_list(self):
        self.next_task_list = None
        self.next_task_list_idx = 0

    def list_tasks(self, cur_time):
        logging.info(f"LIST_TASK {self.client_end_time_map} || {self.client_fake_end_time_map}")
        if self.next_task_list is None:
            self.next_task_list = []
            for k, v in self.client_end_time_map.items():
                if v <= cur_time:
                    self.next_task_list.append(k)

        return self.next_task_list

    def select_participants(self, available_clients, cur_time):
        clients_not_busy = []
        for client in available_clients:
            busy = False
            if client in self.client_end_time_map:
                end_time = self.client_end_time_map[client]
                if end_time >= cur_time:
                    busy = True
            elif client in self.client_fake_end_time_map:
                end_time = self.client_fake_end_time_map[client]
                if end_time >= cur_time:
                    busy = True

            if not busy:
                clients_not_busy.append(client)
        available_clients = clients_not_busy

        if self.mode in ["async", "local"]:
            self.rng.shuffle(available_clients)
            picked_clients = available_clients[:self.async_num_issues_max]

        return sorted(picked_clients)

    def register_end_time(self, client, end_time):
        # better have end_time to be int
        self.client_end_time_map[client] = end_time

    def register_fake_end_time(self, client, fake_end_time):
        # better have fake_end_time to be int
        self.client_fake_end_time_map[client] = fake_end_time

    def refresh_record(self, cur_time):
        # because we must have processed them before
        # I mean, through get_next_task in outer caller's training_handler()

        del_key = []
        for k, v in self.client_end_time_map.items():
            if v <= cur_time:
                del_key.append(k)
        for k in del_key:
            del self.client_end_time_map[k]
        del_key = []
        for k, v in self.client_fake_end_time_map.items():
            if v <= cur_time:
                del_key.append(k)
        for k in del_key:
            del self.client_fake_end_time_map[k]