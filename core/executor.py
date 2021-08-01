# -*- coding: utf-8 -*-
from fl_client_libs import *
from argparse import Namespace
import gc
from client import Client


class Executor(object):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""
    def __init__(self, args):

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')
        self.executors = [int(v) for v in str(args.learners).split('-')]

        # ======== env information ========
        self.this_rank = args.this_rank

        # ======== model and data ========
        self.model = self.training_sets = self.test_dataset = None
        self.temp_model_path = os.path.join(logDir, 'model_'+str(args.this_rank)+'.pth.tar')

        # ======== channels ========
        self.server_event_queue = self.client_event_queue = None
        self.control_manager = None

        # ======== runtime information ========
        self.collate_fn = None
        self.task = args.task
        self.epoch = 0
        self.start_run_time = time.time()

        # JZF
        self.test_mode = args.test_mode
        self.sample_mode = args.sample_mode
        self.filter_less = args.filter_less
        self.filter_more = args.filter_more
        self.batch_size = args.batch_size
        self.personalized = args.personalized

    def setup_env(self):
        logging.info(f"(EXECUTOR:{self.this_rank}) is setting up environ ...")

        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda and self.device == None:
            for i in range(torch.cuda.device_count()):
                try:
                    self.device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    print(torch.rand(1).to(device=self.device))
                    logging.info(f'End up with cuda device ({self.device})')
                    break
                except Exception as e:
                    assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'

        self.init_control_communication(self.args.ps_ip, self.args.manager_port)
        self.init_data_communication()


    def setup_seed(self, seed=1):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self, ps_ip, ps_port):
        # Create communication channel between aggregator and worker
        # This channel serves control messages

        logging.info(f"Start to connect to {ps_ip}:{ps_port} for control plane communication ...")

        #for executor_id in self.executors:
        BaseManager.register('get_server_event_que'+str(self.this_rank))
        BaseManager.register('get_client_event')

        self.control_manager = BaseManager(address=(ps_ip, ps_port), authkey=b'FLPerf')
        start_time, is_connected = time.time(), False

        while time.time() - start_time < 15 and not is_connected:
            try:
                self.control_manager.connect()
                is_connected = True
            except Exception as e:
                time.sleep(numpy.random.rand(1)[0]*5+0.1)
                pass

        assert is_connected, 'Failed to connect to the aggregator'


        self.server_event_queue = eval('self.control_manager.get_server_event_que'+str(self.this_rank)+'()')
        self.client_event_queue = self.control_manager.get_client_event()


    def init_data_communication(self):
        dist.init_process_group(self.args.backend, rank=self.this_rank, world_size=len(self.executors) + 1)


    def init_model(self):
        """Return the model architecture used in training"""
        return init_model()

    def init_data(self):
        """Return the training and testing dataset"""
        train_dataset, test_dataset = init_dataset(self.filter_less, self.filter_more)

        # load data partitioner (entire_train_data)
        logging.info("Data partitioner starts ...")

        training_sets = DataPartitioner(data=train_dataset, numOfClass=self.args.num_class)
        training_sets.partition_data_helper(num_clients=self.args.total_worker,
                                                data_map_file=self.args.data_map_file)

        testing_sets = DataPartitioner(data=test_dataset, numOfClass=self.args.num_class, isTest=True)
        testing_sets.partition_data_helper(num_clients=len(self.executors))

        logging.info("Data partitioner completes ...")


        if self.task == 'nlp':
            self.collate_fn = collate
        elif self.task == 'voice':
            self.collate_fn = voice_collate_fn

        return training_sets, testing_sets

    def init_data_centralized_train(self):
        centralized_train_dataset, _ = init_dataset(self.filter_less, self.filter_more)

        centralized_training_sets = DataPartitioner(data=centralized_train_dataset, numOfClass=self.args.num_class)
        centralized_training_sets.partition_data_helper(num_clients=1)

        return centralized_training_sets

    def init_data_all_test(self):
        all_test_dataset = init_dataset_all_test()
        all_testing_sets = DataPartitioner(data=all_test_dataset, numOfClass=self.args.num_class, isTest=True)
        all_testing_sets.partition_data_helper(num_clients=self.args.total_worker, data_map_file=self.args.data_map_file)
        return all_testing_sets

    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.model = self.model.to(device=self.device)
        self.training_sets, self.testing_sets = self.init_data()
        if self.test_mode == "all":
            self.all_testing_sets = self.init_data_all_test()
        if self.sample_mode == "centralized":
            self.centralized_training_sets = self.init_data_centralized_train()
        self.start_event()
        self.event_monitor()

    def start_event(self):
        executor_info = self.report_executor_info_handler()
        if self.sample_mode == "centralized" and self.this_rank == 1:  # related to centralized training
            client_data = select_dataset(1, self.centralized_training_sets, batch_size=self.batch_size,
                                         collate_fn=self.collate_fn)
            executor_info['global_num_batches'] = len(client_data)
        self.push_msg_to_server('report_executor_info', executor_info)

    def push_msg_to_server(self, event, results):
        self.client_event_queue.put({'return': results, 'event': event, 'executorId': self.this_rank})


    def push_msg_to_server_asyn(self, event, results):
        self.client_event_queue.put_nowait({'return': results, 'event': event, 'executorId': self.this_rank})


    def report_executor_info_handler(self):
        """Return the statistics of training dataset"""
        return self.training_sets.getSize()


    def update_model_handler(self):
        self.epoch += 1

        # self.model = self.model.to(device='cpu')

        # waiting_list = []
        """Update the model copy on this executor"""
        for param in self.model.parameters():
            temp_tensor = torch.zeros_like(param.data, device='cpu')
            dist.recv(tensor=temp_tensor, src=0)
            #req = dist.irecv(tensor=param.data, src=0)
            param.data = temp_tensor.to(device=self.device)
        #     waiting_list.append(req)

        # for req in waiting_list:
        #     req.wait()

        # self.model = self.model.to(device=self.device)

        # Dump model every dumping interval

        if self.epoch % self.args.dump_epoch == 0 and self.this_rank == 1:
            with open(self.temp_model_path+'_'+str(self.epoch), 'wb') as model_out:
                pickle.dump(self.model, model_out)

        # Dump latest model to disk
        with open(self.temp_model_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)


    def load_global_model(self):
        # load last global model
        with open(self.temp_model_path, 'rb') as model_in:
            model = pickle.load(model_in)
        return model


    def override_conf(self, config):
        default_conf = vars(self.args).copy()

        for key in config:
            default_conf[key] = config[key]

        return Namespace(**default_conf)


    def get_client_trainer(self, conf):
        """Developer can redefine to this function to customize the training:
           API:
            - train(client_data=client_data, model=client_model, conf=conf)
        """
        return Client(conf)


    def training_handler(self, clientId, conf):
        """Train model given client ids"""

        # load last global model
        client_model = self.load_global_model()

        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer

        if clientId == 0: # related to centralized training
            client_data = select_dataset(1, self.centralized_training_sets, batch_size=conf.batch_size,
                                         collate_fn=self.collate_fn)
        else:
            client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size,
                                         collate_fn=self.collate_fn)

        client = self.get_client_trainer(conf)
        train_res = client.train(client_data=client_data, model=client_model, conf=conf)

        # we need to get runtime variance for BN
        self.model = client_model
        return train_res


    def testing_handler(self, args):
        """Test model"""
        evalStart = time.time()
        device = self.device
        data_loader = select_dataset(self.this_rank, self.testing_sets, batch_size=args.test_bsz, isTest=True, collate_fn=self.collate_fn)

        if self.task == 'voice':
            criterion = CTCLoss(reduction='mean').to(device=device)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(device=device)

        test_res = test_model(self.this_rank, self.model, data_loader, device=device, criterion=criterion, tokenizer=tokenizer)

        test_loss, acc, acc_5, testResults = test_res
        logging.info("After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                    .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

        gc.collect()

        return testResults

    def all_testing_handler(self, clientId, conf):
        """Test model"""
        evalStart = time.time()

        if self.personalized == "meta": # one step forward
            # load last global model
            client_model = self.load_global_model()

            conf.clientId, conf.device = clientId, self.device
            conf.tokenizer = tokenizer

            client_data = select_dataset(clientId, self.training_sets, batch_size=conf.batch_size,
                                         collate_fn=self.collate_fn)
            client = self.get_client_trainer(conf)
            _ = client.train(client_data=client_data, model=client_model, conf=conf,
                             specified_loop_num=1) # for "meta"
            self.model = client_model

        device = self.device
        data_loader = select_dataset(clientId, self.all_testing_sets, batch_size=args.test_bsz, isTest=True,
                                     collate_fn=self.collate_fn)

        if self.task == 'voice':
            criterion = CTCLoss(reduction='mean').to(device=device)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(device=device)

        all_test_res = test_model(clientId, self.model, data_loader, device=device, criterion=criterion,
                              tokenizer=tokenizer)

        test_loss, acc, acc_5, testResults = all_test_res
        logging.info(
            "After aggregation epoch {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
            .format(self.epoch, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4),
                    test_loss, acc * 100., acc_5 * 100.))

        gc.collect()

        return testResults

    def event_monitor(self):
        """Activate event handler once receiving new message"""
        logging.info("Start monitoring events ...")

        while True:
            if not self.server_event_queue.empty():
                event_dict = self.server_event_queue.get()
                event_msg = event_dict['event']

                logging.info(f"Executor {self.this_rank}: Received (Event:{event_msg.upper()}) from aggregator")

                if event_msg == 'report_executor_info':
                    executor_info = self.report_executor_info_handler()
                    self.push_msg_to_server(event_msg, executor_info)

                elif event_msg == 'update_model':
                    self.update_model_handler()

                # initiate each training round
                elif event_msg == 'train':
                    clientId, client_conf = event_dict['clientId'], self.override_conf(event_dict['conf'])

                    train_res = self.training_handler(clientId=clientId, conf=client_conf)
                    self.push_msg_to_server('train_nowait', None)
                    # model updates may be time-consuming, thus we apply asyn push for better communication-computation overlaps
                    self.push_msg_to_server_asyn(event_msg, train_res)

                elif event_msg == 'test':
                    test_res = self.testing_handler(args=self.args)
                    self.push_msg_to_server(event_msg, test_res)

                elif event_msg == "all_test":
                    clientId, client_conf = event_dict['clientId'], self.override_conf(event_dict['conf'])

                    all_test_res = self.all_testing_handler(clientId=clientId, conf=client_conf)
                    self.push_msg_to_server('all_test_nowait', None)
                    self.push_msg_to_server_asyn(event_msg, all_test_res)

                elif event_msg == 'stop':
                    self.stop()
                    break

                else:
                    logging.error("Unknown message types!")

                time.sleep(0.3)


    def stop(self):
        logging.info(f"Terminating (Executor {self.this_rank}) ...")

        #self.control_manager.shutdown()


if __name__ == "__main__":
    executor = Executor(args)
    executor.run()


