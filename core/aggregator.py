# -*- coding: utf-8 -*-

from fl_aggregator_libs import *
from random import Random
from resource_manager import ResourceManager
from async_controller import AsyncController

class Aggregator(object):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.executors = [int(v) for v in str(args.learners).split('-')]
        self.num_executors = len(self.executors)

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.resource_manager = ResourceManager()
        self.client_manager = self.init_client_manager(args=args)

        # ======== model and data ========
        self.model = None

        # list of parameters in model.parameters()
        self.model_in_update = []
        self.last_global_model = []

        # ======== channels ========
        self.server_event_queue = {}
        self.client_event_queue = Queue()
        self.control_manager = None
        # event queue of its own functions
        self.event_queue = collections.deque()

        # ======== runtime information ========
        self.tasks_round = 0
        self.sampled_participants = []

        self.round_stragglers = []
        self.model_update_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.epoch = 0

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # === JZF ===
        self.test_mode = self.args.test_mode
        self.sample_mode = self.args.sample_mode
        self.global_num_batches = None
        self.sync_mode = self.args.sync_mode
        self.global_model_has_changed = False
        if self.sync_mode in ["async"]:
            self.async_controller = AsyncController(args)
            self.async_sec_per_step = args.async_sec_per_step
            self.async_step = 0 # then will start from 0
            self.async_eval_interval = args.async_eval_interval
            self.async_end_time = args.async_end_time
            self.async_need_broadcast_model = False
            self.async_total_num_steps = self.async_end_time // self.async_sec_per_step
            self.async_update_ratio = args.async_update_ratio

        # number of registered executors
        self.registered_executor_info = 0
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

        if self.test_mode == "all":
            self.local_testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                                    'gradient_policy': args.gradient_policy, 'task': args.task,
                                    'perf': collections.OrderedDict()}

        self.gradient_controller = None
        if self.args.gradient_policy == 'yogi':
            from utils.yogi import YoGi
            self.gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)

        # ======== Task specific ============
        self.imdb = None           # object detection

    def setup_env(self):
        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda and self.device == None:
            for i in range(torch.cuda.device_count()):
                if i < self.args.aggregator_device: # skip some gpus
                    continue
                try:
                    self.device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    _ = torch.rand(1).to(device=self.device)
                    logging.info(f'End up with cuda device ({self.device})')
                    break
                except Exception as e:
                    assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'

        self.init_control_communication(self.args.ps_ip, self.args.manager_port, self.executors)
        self.init_data_communication()


    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self, ps_ip, ps_port, executors):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Start to initiate {ps_ip}:{ps_port} for control plane communication ...")

        dummy_que = {executorId:Queue() for executorId in executors}
        # create multiple queue for each aggregator_executor pair
        for executorId in executors:
            BaseManager.register('get_server_event_que'+str(executorId), callable=lambda: dummy_que[executorId])

        dummy_client_que = Queue()
        BaseManager.register('get_client_event', callable=lambda: dummy_client_que)

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        self.control_manager.start()

        #self.server_event_queue = torch.multiprocessing.Manager().dict()
        for executorId in self.executors:
            self.server_event_queue[executorId] = eval('self.control_manager.get_server_event_que'+str(executorId)+'()')

        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)


    def init_model(self):
        """Load model"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

        return init_model()

    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Kuiper sampler
                - Kuiper prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://arxiv.org/abs/2010.06081
        """

        # sample_mode: random or kuiper
        client_manager = clientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        # load client profiles
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def executor_info_handler(self, executorId, info):
        if self.sample_mode == "centralized" and executorId == 1:
            self.global_num_batches = info['global_num_batches']

        self.registered_executor_info += 1

        # have collected all executors
        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout

        if self.registered_executor_info == self.num_executors:

            if self.sample_mode == "centralized":
                clientId = 0
                total_size = 0
                for _, size in enumerate(info['size']):
                    if size >= self.client_manager.filter_less and size <= self.client_manager.filter_more:
                        total_size += size

                # simulate a cloud machine
                systemProfile = {'computation': 1.0, 'communication': 1.25e8} # PCIe 3 / 2
                self.client_manager.registerClient(executorId, clientId,
                                                   size=total_size, speed=systemProfile)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                                                     upload_epoch=self.args.local_steps,
                                                     upload_size=self.model_update_size,
                                                     download_size=self.model_update_size)

            # even if in centralized mode we still need to record the feasibility of separate clients
            # as needed in testing (we only use feasible clients to test)
            clientId = 1

            for index, _size in enumerate(info['size']):
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = clientId%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})

                self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                    upload_epoch=self.args.local_steps, upload_size=self.model_update_size, download_size=self.model_update_size)

                clientId += 1

            logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))

            # start to sample clients
            if self.sync_mode == "async":
                self.async_step_completion_handler()
            else:
                self.round_completion_handler()


    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """We try to remove dummy events as much as possible, by removing the stragglers/offline clients in overcommitment"""

        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            client_cfg = self.client_conf.get(client_to_run, self.args)

            if self.sample_mode == "centralized":
                true_upload_epoch = self.global_num_batches
            else:
                true_upload_epoch = client_cfg.local_steps
            exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                    batch_size=client_cfg.batch_size, upload_epoch=true_upload_epoch,
                                    upload_size=self.model_update_size, download_size=self.model_update_size)

            roundDuration = exe_cost['computation'] + exe_cost['communication']
            # if the client is not active by the time of collection, we consider it is lost in this round
            if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = exe_cost

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
        round_duration = completionTimes[top_k_index[-1]]

        return clients_to_run, dummy_clients, completed_client_clock, round_duration


    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.save_last_param()

        self.model_update_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
        self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)
        self.start_event()
        self.event_monitor()


    def start_event(self):
        #self.model_in_update = [param.data for idx, param in enumerate(self.model.parameters())]
        #self.event_queue.append('report_executor_info')
        pass

    def broadcast_msg(self, msg):
        for executorId in self.executors:
            self.server_event_queue[executorId].put_nowait(msg)


    def broadcast_models(self):
        """Push the latest model to executors"""
        # self.model = self.model.to(device='cpu')

        # waiting_list = []
        for param in self.model.parameters():
            temp_tensor = param.data.to(device='cpu')
            for executorId in self.executors:
                dist.send(tensor=temp_tensor, dst=executorId)
                # req = dist.isend(tensor=param.data, dst=executorId)
                # waiting_list.append(req)

        # for req in waiting_list:
        #     req.wait()

        # self.model = self.model.to(device=self.device)


    def select_participants(self, select_num_participants, overcommitment=1.3):
        if self.sample_mode == "centralized":
            return [0]
        else:
            return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment),
                                                              cur_time=self.global_virtual_clock))


    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        if self.args.gradient_policy in ['qfedavg']:
            self.client_training_results.append(results)

        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
                    time_stamp=self.epoch,
                    duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['communication']
                )

        device = self.device
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        importance = 1./self.tasks_round
        if len(self.model_in_update) == 0:
            self.model_in_update = [True]

            for idx, param in enumerate(self.model.parameters()):
                param.data = torch.from_numpy(results['update_weight'][idx]).to(device=device)*importance
        else:
            for idx, param in enumerate(self.model.parameters()):
                param.data += torch.from_numpy(results['update_weight'][idx]).to(device=device)*importance


    def save_last_param(self):
        self.last_global_model = [param.data.clone() for param in self.model.parameters()]

    def round_weight_handler(self, last_model, current_model):
        if self.epoch > 1:

            if self.args.gradient_policy == 'yogi':
                last_model = [x.to(device=self.device) for x in last_model]
                current_model = [x.to(device=self.device) for x in current_model]

                diff_weight = self.gradient_controller.update([pb-pa for pa, pb in zip(last_model, current_model)])

                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] + diff_weight[idx]

            elif self.args.gradient_policy == 'qfedavg':

                learning_rate, qfedq = self.args.learning_rate, self.args.qfed_q
                Deltas, hs = None, 0.
                last_model = [x.to(device=self.device) for x in last_model]

                for result in self.client_training_results:
                    # plug in the weight updates into the gradient
                    grads = [(u - torch.from_numpy(v).to(device=self.device)) * 1.0 / learning_rate for u, v in zip(last_model, result['update_weight'])]
                    loss = result['moving_loss']

                    if Deltas is None:
                        Deltas = [np.float_power(loss+1e-10, qfedq) * grad for grad in grads]
                    else:
                        for idx in range(len(Deltas)):
                            Deltas[idx] += np.float_power(loss+1e-10, qfedq) * grads[idx]

                    # estimation of the local Lipchitz constant
                    hs += (qfedq * np.float_power(loss+1e-10, (qfedq-1)) * torch.sum(torch.stack([torch.square(grad).sum() for grad in grads])) + (1.0/learning_rate) * np.float_power(loss+1e-10, qfedq))

                # update global model
                for idx, param in enumerate(self.model.parameters()):
                    param.data = last_model[idx] - Deltas[idx]/(hs+1e-10)

    def async_client_completion_handler(self, results):
        if not self.async_need_broadcast_model:
            self.async_need_broadcast_model = True

        self.loss_accumulator.append(results['moving_loss'])
        self.stats_util_accumulator.append(results['utility'])

        device = self.device
        importance = 1. / self.tasks_round
        if len(self.model_in_update) == 0:
            self.model_in_update = [True]

            for idx, param in enumerate(self.model.parameters()):
                remaining_ratio = 1 - self.async_update_ratio
                if idx == 0:
                    logging.info(f"{importance} {self.async_update_ratio} {remaining_ratio}")
                    logging.info(f"Before: {param.data.cpu().numpy().flatten()[:10]}")
                param.data *= remaining_ratio
                param.data += self.async_update_ratio *\
                             torch.from_numpy(results['update_weight'][idx]).to(device=device) * importance
                if idx == 0:
                    logging.info(f"After: {param.data.cpu().numpy().flatten()[:10]}")
        else:
            for idx, param in enumerate(self.model.parameters()):
                param.data += self.async_update_ratio *\
                              torch.from_numpy(results['update_weight'][idx]).to(device=device) * importance

    def async_step_completion_handler(self):
        self.resource_manager.register_all_test_tasks(self.client_manager.feasibleClients)
        self.async_controller.refresh_record(self.global_virtual_clock)
        self.async_controller.refresh_next_task_list()

        # update the information on clients that should start at this time step
        available_participants = self.select_participants(select_num_participants=100000000)
        async_sampled_clients = self.async_controller.select_participants(available_participants,
                                                                          self.global_virtual_clock)

        final_participated_clients = []
        lost_clients = []
        for client_to_run in async_sampled_clients:
            client_cfg = self.client_conf.get(client_to_run, self.args)
            exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                    batch_size=client_cfg.batch_size, upload_epoch=client_cfg.local_steps,
                                    upload_size=self.model_update_size, download_size=self.model_update_size)

            roundDuration = exe_cost['computation'] + exe_cost['communication']
            # if the client is not active by the time of collection, we consider it is lost in this round
            end_time = round(roundDuration + self.global_virtual_clock) # better have end_time to be int
            active = True
            # actually sync mode should also do such a fine-grained check
            # but the sanity of async is just more sensitive to it
            if end_time > self.global_virtual_clock:
                for i in range(round(self.global_virtual_clock), end_time + 1):
                    if not self.client_manager.isClientActive(client_to_run, i):
                        active = False
                        break
            if active:
                final_participated_clients.append(client_to_run)
                self.async_controller.register_end_time(client_to_run, end_time)
            else:
                lost_clients.append(client_to_run)
                # the clients go here are in two possible situations
                # 1. not online at the moment at all
                # 2. online, but just doing some training task
                # that it will abort afterwards in the future
                # in this case, it is still not gonna to be used
                self.async_controller.register_fake_end_time(client_to_run, end_time)

        logging.info(f"[Async] {len(final_participated_clients)} clients to start "
                     f"at step {self.async_step}/{self.async_total_num_steps} "
                     f"wall clock {round(self.global_virtual_clock)} s: {final_participated_clients}")
        logging.info(f"[Async] while {len(lost_clients)}/{len(async_sampled_clients)} sampled clients "
                     f"dropped or busy: {lost_clients}")

        if len(final_participated_clients) > 0:
            self.broadcast_msg({
                'event': 'update_async_model',
                'clientIds': final_participated_clients
            })  # clients should use the appropriate version of global model
            self.broadcast_models()

        # move on
        self.global_virtual_clock += self.async_sec_per_step
        self.async_step += 1

        if self.global_virtual_clock >= self.async_end_time:
            self.event_queue.append('stop')
        else:
            # if self.async_step == 1:
            #     self.broadcast_msg({'event': 'update_model'})
            #     self.broadcast_models()
            self.event_queue.append('start_round')

    def round_completion_handler(self):
        self.global_virtual_clock += self.round_duration
        self.epoch += 1

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_global_model, [param.data.clone() for param in self.model.parameters()])

        avgUtilLastEpoch = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.registerScore(clientId, avgUtilLastEpoch,
                                    time_stamp=self.epoch,
                                    duration=self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['communication'],
                                    success=False)

        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, Epoch: {self.epoch}, Planned participants: " + \
            f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # update select participants
        self.sampled_participants = self.select_participants(select_num_participants=self.args.total_worker, overcommitment=self.args.overcommitment)
        clientsToRun, round_stragglers, virtual_client_clock, round_duration = self.tictak_client_tasks(self.sampled_participants, self.args.total_worker)

        logging.info(f"Selected participants to run: {clientsToRun}:\n{virtual_client_clock}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)

        all_test_clients = self.client_manager.feasibleClients
        self.resource_manager.register_all_test_tasks(all_test_clients)

        self.tasks_round = len(clientsToRun)

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.round_duration = round_duration
        self.model_in_update = []
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.epoch >= self.args.epochs:
            self.event_queue.append('stop')
        elif self.epoch % self.args.eval_interval == 0:
            self.event_queue.append('update_model')
            self.event_queue.append('test')
        else:
            self.event_queue.append('update_model')
            self.event_queue.append('start_round')


    def testing_completion_handler(self, results):
        self.test_result_accumulator.append(results)

        if self.test_mode == "all":
            current = len(self.test_result_accumulator)
            all = len(self.client_manager.feasibleClients)
            ten_cent = max(1, all // 10)
            if current % ten_cent == 0:
                logging.info(f"Testing progress: {current}/{all} clients ({current // ten_cent * 10}%)")

        if self.test_mode == "all" and len(self.test_result_accumulator) == len(self.client_manager.feasibleClients):
            pass
        elif self.test_mode == "default" and len(self.test_result_accumulator) == len(self.executors):
            pass
        else:
            return

        # Have collected all testing results
        accumulator = self.test_result_accumulator[0]
        for i in range(1, len(self.test_result_accumulator)):
            if self.args.task == "detection":
                for key in accumulator:
                    if key == "boxes":
                        for j in range(self.imdb.num_classes):
                            accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                    else:
                        accumulator[key] += self.test_result_accumulator[i][key]
            else:
                for key in accumulator:
                    accumulator[key] += self.test_result_accumulator[i][key]
        if self.args.task == "detection":
            self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                'loss': accumulator['test_loss'],
                'test_len': accumulator['test_len']
                }
        else:
            self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                'loss': accumulator['test_loss']/accumulator['test_len'],
                'test_len': accumulator['test_len']
                }


        logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['loss'],
                self.testing_history['perf'][self.epoch]['test_len']))

        if self.test_mode == "all":
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch].update({
                    'local_top_1': round(accumulator['local_top_1'] * 100.0 / len(self.test_result_accumulator), 4),
                    'local_top_5': round(accumulator['local_top_5'] * 100.0 / len(self.test_result_accumulator), 4),
                    'local_loss': accumulator['local_test_loss'],
                    'local_test_len': accumulator['local_test_len']
                })
            else:
                self.testing_history['perf'][self.epoch].update({
                    'local_top_1': round(accumulator['local_top_1'] / accumulator['local_test_len'] * 100.0, 4),
                    'local_top_5': round(accumulator['local_top_5'] / accumulator['local_test_len'] * 100.0, 4),
                    'local_loss': accumulator['local_test_loss'] / accumulator['local_test_len'],
                    'local_test_len': accumulator['local_test_len']
                })

            logging.info(
                "FL Local Testing in epoch: {}, virtual_clock: {}, top_1: {} %, "
                "top_5: {} %, test loss: {:.4f}, test len: {}"
                .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['local_top_1'],
                        self.testing_history['perf'][self.epoch]['local_top_5'],
                        self.testing_history['perf'][self.epoch]['local_loss'],
                        self.testing_history['perf'][self.epoch]['local_test_len']))

        if self.sync_mode in ["async"]:
            self.test_result_accumulator = []
            self.async_step_completion_handler()
        else:
            self.event_queue.append('start_round')

    def get_client_conf(self, clientId):
        # learning rate scheduler
        conf = {}
        conf['learning_rate'] = self.args.learning_rate
        return conf

    def event_monitor(self):
        logging.info("Start monitoring events ...")

        if self.sync_mode == "async":
            while True:
                if len(self.event_queue) != 0:
                    event_msg = self.event_queue.popleft()
                    send_msg = {'event': event_msg}

                    if event_msg == 'start_round':
                        tasks = self.async_controller.list_tasks(self.global_virtual_clock)
                        self.tasks_round = len(tasks)
                        if self.tasks_round == 0:
                            self.event_queue.append('test')
                        else:
                            logging.info(f"[Async] {self.tasks_round} clients to end "
                                         f"at step {self.async_step}/{self.async_total_num_steps} "
                                         f"wall clock {round(self.global_virtual_clock)} s: {tasks}")

                        for executorId in self.executors:
                            next_clientId = self.async_controller.get_next_task(self.global_virtual_clock)
                            if next_clientId is not None:
                                config = self.get_client_conf(next_clientId)
                                self.server_event_queue[executorId].put(
                                    {'event': 'train', 'clientId': next_clientId, 'conf': config})
                    elif event_msg == 'stop':
                        self.broadcast_msg(send_msg)
                        self.stop()
                        break
                    elif event_msg == 'test':
                        if self.global_virtual_clock % self.async_eval_interval == 0:
                            self.broadcast_msg({'event': 'update_model'})  # all use the latest global model
                            self.broadcast_models()
                            if self.test_mode == "all":
                                for executorId in self.executors:
                                    next_clientId = self.resource_manager.get_next_all_test_task()

                                    if next_clientId is not None:
                                        config = self.get_client_conf(next_clientId)
                                        self.server_event_queue[executorId].put(
                                            {'event': 'test', 'clientId': next_clientId, 'conf': config}
                                        )
                            else:
                                self.broadcast_msg(send_msg)
                        else:
                            self.async_step_completion_handler()

                elif not self.client_event_queue.empty():
                    event_dict = self.client_event_queue.get()
                    event_msg, executorId, results = event_dict['event'], event_dict['executorId'], event_dict['return']

                    if self.sync_mode in ["async"]:
                        logging.info(f"Step {self.async_step}: Receive (Event:{event_msg.upper()})"
                                     f" from (Executor:{executorId})")
                    else:
                        logging.info(f"Round {self.epoch}: Receive (Event:{event_msg.upper()})"
                                     f" from (Executor:{executorId})")

                    if event_msg == 'report_executor_info':
                        self.executor_info_handler(executorId, results)
                    elif event_msg == 'train':
                        self.async_client_completion_handler(results)
                        logging.info(f"[Async] {len(self.stats_util_accumulator)}/{self.tasks_round}")
                        if len(self.stats_util_accumulator) == self.tasks_round:
                            avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
                            logging.info(f"[Async] All {self.tasks_round} clients at {self.global_virtual_clock} s end; "
                                         f"moving loss all the way around: {avg_loss}")
                            self.stats_util_accumulator = []
                            self.model_in_update = []
                            self.save_last_param()
                            self.event_queue.append("test")
                    elif event_msg == 'test':
                        self.testing_completion_handler(results)
                    elif event_msg == 'train_nowait':
                        next_clientId = self.async_controller.get_next_task(self.global_virtual_clock)
                        if next_clientId is not None:
                            config = self.get_client_conf(next_clientId)
                            runtime_profile = {'event': 'train', 'clientId': next_clientId, 'conf': config}
                            self.server_event_queue[executorId].put(runtime_profile)
                    elif event_msg == 'test_nowait':
                        if self.test_mode == "all":
                            next_clientId = self.resource_manager.get_next_all_test_task()
                            if next_clientId is not None:
                                config = self.get_client_conf(next_clientId)
                                runtime_profile = {'event': 'test', 'clientId': next_clientId, 'conf': config,
                                                   'necessary': False}
                                self.server_event_queue[executorId].put(runtime_profile)
                        else:
                            pass
                    else:
                        logging.error(f"Unknown message types: {event_msg}")

                # execute every 100 ms
                time.sleep(0.1)
        else:
            while True:
                if len(self.event_queue) != 0:
                    event_msg = self.event_queue.popleft()
                    send_msg = {'event': event_msg}

                    if event_msg == 'update_model':
                        self.broadcast_msg(send_msg)
                        self.broadcast_models()

                    elif event_msg == 'start_round':
                        if self.sample_mode == "centralized":
                            config = self.get_client_conf(0)
                            self.server_event_queue[1].put( # note that executors start numbering from 1
                                {'event': 'train', 'clientId': 0, 'conf': config})
                        else:
                            for executorId in self.executors:
                                next_clientId = self.resource_manager.get_next_task()
                                if next_clientId is not None:
                                    config = self.get_client_conf(next_clientId)
                                    self.server_event_queue[executorId].put({'event': 'train', 'clientId':next_clientId, 'conf': config})

                    elif event_msg == 'stop':
                        self.broadcast_msg(send_msg)
                        self.stop()
                        break

                    elif event_msg == 'report_executor_info':
                        self.broadcast_msg(send_msg)

                    elif event_msg == 'test':
                        if self.test_mode == "all":
                            for executorId in self.executors:
                                next_clientId = self.resource_manager.get_next_all_test_task()

                                if next_clientId is not None:
                                    config = self.get_client_conf(next_clientId)
                                    self.server_event_queue[executorId].put(
                                        {'event': 'test', 'clientId': next_clientId, 'conf': config}
                                    )
                        else:
                            self.broadcast_msg(send_msg)

                elif not self.client_event_queue.empty():

                    event_dict = self.client_event_queue.get()
                    event_msg, executorId, results = event_dict['event'], event_dict['executorId'], event_dict['return']

                    if event_msg != 'train_nowait' and event_msg != 'test_nowait':
                        logging.info(f"Round {self.epoch}: Receive (Event:{event_msg.upper()}) from (Executor:{executorId})")

                    # collect training returns from the executor
                    if event_msg == 'train_nowait':
                        # pop a new client to run
                        if self.sample_mode == "centralized":
                            pass
                        else:
                            next_clientId = self.resource_manager.get_next_task()

                            if next_clientId is not None:
                                config = self.get_client_conf(next_clientId)
                                runtime_profile = {'event': 'train', 'clientId':next_clientId, 'conf': config}
                                self.server_event_queue[executorId].put(runtime_profile)

                    elif event_msg == 'test_nowait':
                        if self.test_mode == "all":
                            next_clientId = self.resource_manager.get_next_all_test_task()
                            if next_clientId is not None:
                                config = self.get_client_conf(next_clientId)
                                runtime_profile = {'event': 'test', 'clientId': next_clientId, 'conf': config,
                                                   'necessary': False}
                                self.server_event_queue[executorId].put(runtime_profile)
                        else:
                            pass

                    elif event_msg == 'train':
                        # push training results
                        self.client_completion_handler(results)

                        if len(self.stats_util_accumulator) == self.tasks_round:
                            self.round_completion_handler()

                    elif event_msg == 'test':
                        self.testing_completion_handler(results)

                    elif event_msg == 'report_executor_info':
                        self.executor_info_handler(executorId, results)
                    else:
                        logging.error(f"Unknown message types: {event_msg}")

                # execute every 100 ms
                time.sleep(0.1)


    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)
        self.control_manager.shutdown()

if __name__ == "__main__":
    aggregator = Aggregator(args)
    aggregator.run()

