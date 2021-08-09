import torch
import logging
import math, numpy
from utils.nlp import mask_tokens
from torch.autograd import Variable
import copy
import gc

class Client(object):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        pass

    def train(self, client_data, model, conf, specified_local_steps=None, client_model=None):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)
        if conf.gradient_policy == 'prox':
            global_model = [param.data.clone() for param in model.parameters()]

        if conf.task == "detection":
            lr = conf.learning_rate
            params = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        elif conf.task == 'nlp':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": conf.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=conf.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)

        if conf.task == 'voice':
            from torch_baidu_ctc import CTCLoss
            criterion = CTCLoss(reduction='none').to(device=device)
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0

        if conf.task == "detection":
            im_data = Variable(torch.FloatTensor(1).cuda())
            im_info = Variable(torch.FloatTensor(1).cuda())
            num_boxes = Variable(torch.LongTensor(1).cuda())
            gt_boxes = Variable(torch.FloatTensor(1).cuda())

        # TODO: One may hope to run fixed number of epochs, instead of iterations
        if conf.personalized == "meta":
            if specified_local_steps is not None: # mainly for stepping in testing
                loop_num = 1
            else:
                loop_num = 4 # calculate grad 4 times for each step in training
            delta = 1e-3
            beta = 0.01
        elif conf.personalized == "ditto":
            lam = 3e-1
            loop_num = 2
        else:
            loop_num = 1
        break_while_flag = False

        loader = iter(client_data)
        if conf.sample_mode == "centralized":
            if conf.personalized == "meta" and specified_local_steps is not None: # testing of "meta"
                true_num_steps = specified_local_steps
                start_idx = numpy.random.randint(0, len(client_data))
                for _ in range(start_idx):
                    _ = loader.next()
            else:
                true_num_steps = len(client_data)
            # currently centralized learning goes with 1 epoch

        while True:
            l2_norm_square = 0.0  # for running adapted oort atop "ditto"

            if conf.personalized == "meta" and specified_local_steps is None:
                local_model_copies = []
                for _, param in enumerate(model.parameters()):
                    local_model_copies.append(param.data.detach().clone())

            completed_steps += 1
            for loop_idx in range(loop_num):
                if conf.personalized == "meta":
                    if loop_idx < 3:
                        try:
                            data_pair = loader.next()
                        except StopIteration:
                            loader = iter(client_data)
                            data_pair = loader.next()
                        if loop_idx == 2:
                            data_pair_copy = copy.deepcopy(data_pair)
                    else:
                        data_pair = data_pair_copy
                elif conf.personalized == "ditto":
                    if loop_idx == 0:
                        try:
                            data_pair = loader.next()
                        except StopIteration:
                            loader = iter(client_data)
                            data_pair = loader.next()
                        data_pair_copy = copy.deepcopy(data_pair)
                    else:
                        data_pair = data_pair_copy
                else:
                    try:
                        data_pair = loader.next()
                    except StopIteration:
                        loader = iter(client_data)
                        data_pair = loader.next()

                try:
                    if conf.task == 'nlp':
                        (data, _) = data_pair
                        data, target = mask_tokens(data, tokenizer, conf, device=device)
                    elif conf.task == 'voice':
                        (data, target, input_percentages, target_sizes), _ = data_pair
                        input_sizes = input_percentages.mul_(int(data.size(3))).int()
                    elif conf.task == 'detection':
                        temp_data = data_pair
                        target = temp_data[4]
                        data = temp_data[0:4]
                    else:
                        (data, target) = data_pair

                    if conf.task == "detection":
                        im_data.resize_(data[0].size()).copy_(data[0])
                        im_info.resize_(data[1].size()).copy_(data[1])
                        gt_boxes.resize_(data[2].size()).copy_(data[2])
                        num_boxes.resize_(data[3].size()).copy_(data[3])
                    elif conf.task == 'speech':
                        data = torch.unsqueeze(data, 1).to(device=device)
                    else:
                        data = Variable(data).to(device=device)

                    target = Variable(target).to(device=device)

                    if conf.personalized == "ditto" and loop_idx == 0:
                        true_model = client_model
                    else:
                        true_model = model

                    if conf.task == 'nlp':
                        outputs = true_model(data, labels=target)
                        loss = outputs[0]
                    elif conf.task == 'voice':
                        outputs, output_sizes = true_model(data, input_sizes)
                        outputs = outputs.transpose(0, 1).float()  # TxNxH
                        loss = criterion(outputs, target, output_sizes, target_sizes)
                    elif conf.task == "detection":
                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label = true_model(im_data, im_info, gt_boxes, num_boxes)

                        loss = rpn_loss_cls + rpn_loss_box \
                                + RCNN_loss_cls + RCNN_loss_bbox

                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    else:
                        output = true_model(data)
                        loss = criterion(output, target)

                    # ======== collect training feedback for other decision components [e.g., kuiper selector] ======
                    if conf.task == 'nlp':
                        loss_list = [loss.item()] #[loss.mean().data.item()]
                    elif conf.task == "detection":
                        loss_list = [loss.tolist()]
                        loss = loss.mean()
                    else:
                        loss_list = loss.tolist()
                        loss = loss.mean()

                    if (conf.personalized == "meta" and conf.adaptation_mode == 0
                            and loop_idx == 1 and specified_local_steps is None) \
                            or (conf.personalized == "meta" and conf.adaptation_mode == 1
                            and loop_idx == 0 and specified_local_steps is None) \
                            or (conf.personalized == "ditto" and loop_idx == 1) \
                            or conf.personalized == "none":
                        temp_loss = sum([l**2 for l in loss_list])/float(len(loss_list))

                        if conf.personalized == "ditto":
                            logging.info(f"{clientId} before {completed_steps} {loop_idx} {temp_loss}")
                            temp_loss += 0.5 * lam * l2_norm_square
                            logging.info(f"{clientId} after {completed_steps} {loop_idx} {temp_loss}")

                        # only measure the loss of the first epoch
                        if completed_steps < len(client_data):
                            if epoch_train_loss == 1e-4:
                                epoch_train_loss = temp_loss
                            else:
                                epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss \
                                                   + conf.loss_decay * temp_loss

                    # ========= Define the backward loss ==============
                    # optimizer.zero_grad()
                    true_model.zero_grad()
                    loss.backward()

                    if conf.personalized == "meta" and loop_idx > 0 and specified_local_steps is None:
                        if loop_idx == 1:
                            grad_copies = []
                            for _, param in enumerate(model.parameters()):
                                grad_copies.append(param.grad.clone())

                            dummy_model1 = [(u + delta * v) for u, v in zip(local_model_copies, grad_copies)]
                            for idx, param in enumerate(model.parameters()):
                                param.data = dummy_model1[idx]
                        elif loop_idx == 2:
                            dummy_grad1 = []
                            for _, param in enumerate(model.parameters()):
                                dummy_grad1.append(param.grad.clone())

                            dummy_model2 = [(u - delta * v) for u, v in zip(local_model_copies, grad_copies)]
                            for idx, param in enumerate(model.parameters()):
                                param.data = dummy_model2[idx]
                        else:
                            dummy_grad2 = []
                            for _, param in enumerate(model.parameters()):
                                dummy_grad2.append(param.grad.clone())
                    elif conf.personalized == "ditto" and loop_idx == 0:
                        for param_c, param in zip(client_model.parameters(), model.parameters()):
                            param_c_dataclone = param_c.data.detach().clone()
                            param_dataclone = param.data.detach().clone()
                            difference = param_c_dataclone - param_dataclone
                            eff_grad = param_c.grad.clone() + lam * difference
                            param_c.data -= conf.learning_rate * eff_grad

                            if conf.adaptation_mode == 1:
                                l2_norm_square += torch.square(difference).sum()

                    else:
                        optimizer.step()

                    # ========= Weight handler ========================
                    if not conf.personalized == "meta":  # currently "meta" is not compatible with "prox" in this impl
                        if conf.gradient_policy == 'prox':
                            for idx, param in enumerate(model.parameters()):
                                param.data += conf.learning_rate * conf.proxy_mu * (param.data - global_model[idx])

                except Exception as ex:
                    error_type = ex
                    break_while_flag = True
                    break # break the for loop

            if break_while_flag:
                break # still need to break the while

            if conf.personalized == "meta" and specified_local_steps is None:
                try:
                    correction = [(u - v) / (2 * delta) for u, v in zip(dummy_grad1, dummy_grad2)]
                    for idx, param in enumerate(model.parameters()):
                        param.data = local_model_copies[idx] - beta * grad_copies[idx] \
                               + conf.learning_rate * beta * correction[idx]

                except Exception as ex:
                    error_type = ex
                    break

            if conf.sample_mode == "centralized":
                if completed_steps == true_num_steps:
                    break
            else:
                if completed_steps == conf.local_steps:
                    break

        if specified_local_steps is not None: # is testing in "meta"
            return

        model_param = [param.data.cpu().numpy() for param in model.parameters()]
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(epoch_train_loss)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        if conf.personalized == "meta" and specified_local_steps is None:
            del dummy_grad1
            del dummy_grad2
            del grad_copies
            del local_model_copies
            del client_data
            gc.collect()
            torch.cuda.empty_cache()

        return results


    def test(self, conf):
        pass


