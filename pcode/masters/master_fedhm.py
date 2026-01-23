# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
import torch
import torch.distributed as dist
from pcode.utils.module_state import ModuleState
import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.early_stopping import EarlyStoppingTracker


class MasterFedHM(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients)) #总的客户端id列表 
        self.world_ids = list(range(1, 1 + conf.n_participated))  # 总的参与客户端id列表 

        # create model as well as their corresponding state_dicts.
        # 创建服务器模型 不需要arch
        # _, self.master_model = create_model.define_model( # 获取模型
        #     conf, to_consistent_model=False
        # )
        # 所有客户端的模型架构  
        self.used_client_archs =[ 
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        
        
       
        
        self.conf.used_client_archs = self.used_client_archs
        # 打印日志 客户端使用的模型架构 和服务器已经为客户端创建模型架构
        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")

        #创建客户端模型字典 便于异构聚合 包含了模型参数
        # self.client_models = dict( # 获取客户端模型信息
        #     create_model.define_model(conf, to_consistent_model=False, arch=arch)
        #     for arch in self.used_client_archs
        # )
        # 使用客户端ID作为字典的键
        self.client_models = {
            client_id: create_model.define_model(conf, to_consistent_model=False, client_id=client_id, arch=arch)
            for client_id, arch in zip(range(1, 1 + conf.n_clients), self.used_client_archs)
        }
# 创建服务器模型 选取第一个客户端的全秩模型作为基准
        first_model_name, first_model_instance = self.client_models[1]
        self.master_model = copy.deepcopy(first_model_instance)
        self.master_model.recover_model()


        # 确定每一个客户的具体使用的什么模型 只有名称 没有返回模型参数

        self.clientid2arch = dict( # 获取所有客户端的模型名称
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch

        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        ) # 虽然划分了数据但是全丢了，只保留数据划分器



        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader] # 这里的test_loader和上面的val_loader都是总的，也就是说未平分给客户端的

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")   # 模型的损失函数，这里传入mean表示计算的损失会求平均
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")  # 模型的评估，计算准确率之类
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)   # 记录最佳性能
        if not self.conf.train_fast:
            self.local_coordinator = [create_coordinator.Coordinator(conf, self.metrics) for _ in
                                      range(self.conf.n_clients)]

        conf.logger.log(f"Master initialized the coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )   # 早停装置，为0时关掉早停功能

        # save arguments to disk.
        conf.is_finished = False    # 用于控制全局训练是否已经结束
        checkpoint.save_arguments(conf) # 将超参数保存在json文件中

    def run(self):
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )   # 生成参与数长的列表，每个值为每个客户端的训练轮数
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.对应真实的id号
            selected_client_ids = self._random_select_clients() # 随机选择客户端

            # detect early stopping.
            self._check_early_stopping() # 检查是否满足早停，可能未开启早停

            # init the activation tensor and broadcast to all clients (either start or stop).
            self.activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                self.send_extra_info_to_selected_clients(selected_client_ids)

                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)

            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            self.receive_extra_info_from_selected_clients(selected_client_ids)

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            ) # 接受模型参数

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)   # 聚合模型，评估和存档

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        pass


    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()  # 随机选择客户端
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        msg_len = 3 # 消息行数

        activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids) # 第一行装客户端id
        activation_msg[1, :] = comm_round                       # 第二行装联邦轮次
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)     # 第三行装每个客户端训练轮数

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")

        """
        流程：
        1. 获取 Master 参数 (Full Rank)。
        2. 遍历每个被选中的 Client。
        3. 将 Master 参数加载到 Client 模型中 (此时 Client 模型必须是 Recover 状态才能匹配 Key)。
        4. 根据 Client 配置的 Rank 对模型进行 Decompose。
        5. 打包发送。
        """

# 更新服务器端存储的客户端模型参数（J基于 全局模型参数填充）
        rank_list = self.conf.rank_list # 获取所有模型的秩
        master_model_state_dict= self.master_model.state_dict()#获取 Master 参数

        # for selected_client_id in selected_client_ids:
        #     _,model = self.client_models[selected_client_id] # 取出模型对象
        #     model.to('cuda:7')
    
        #     # 【步骤 1：结构对齐】
 
        #     model.recover_model()
        #     model.to('cuda:7')
    
        #     # 【步骤 2：加载全量参数】
    
        #     model.load_state_dict(master_model_state_dict)

        #     # -----------------------------------------------------------
        #     # 【步骤 3：按需分解】
        #     # 根据该客户端特定的 rank 进行 SVD 分解
        #     # -----------------------------------------------------------
        #     current_rank = rank_list[selected_client_id-1] # 获取该客户端对应的 rank (或 ratio)
            
        #     model.decom_model(current_rank)
        #     model.to('cpu')


        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            _,client_model = self.client_models[selected_client_id] # 1 取出该客户端的模型对象
            client_model.to('cuda:7')
            if hasattr(client_model, "recover_model"):
                client_model.recover_model() # 2 恢复全秩
            client_model.to('cuda:7')

            client_model.load_state_dict(master_model_state_dict) # 3. 加载全局参数

            arch = self.clientid2arch[selected_client_id] # 获取当前循环模型名称

            current_rank_ratio = rank_list[selected_client_id - 1] #根据该客户端的 Rank 进行分解
            if hasattr(client_model, "decom_model"):
                client_model.decom_model(current_rank_ratio)
            client_model.to('cpu')

            # client_model_state_dict = self.client_models[arch].state_dict()
            # 改为 由id获取字典的元组 
            
            client_model_state_dict = client_model.state_dict()


            flatten_model = TensorBuffer(list(client_model_state_dict.values())) # 打包模型参数
            dist.send(tensor=flatten_model.buffer, dst=worker_rank) # 发送参数到对应模型
            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )

        dist.barrier()

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        pass

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids: # 为接受数据准备容器，这也可以解释为什么要在服务器创建模型
            arch = self.clientid2arch[selected_client_id]
            # client_tb = TensorBuffer(
            #     list(self.client_models[arch].state_dict().values())
            # )
            # 改为
            _, client_model = self.client_models[selected_client_id]
            
            current_rank_ratio = self.conf.rank_list[selected_client_id - 1]
            if hasattr(client_model, "decom_model"):
                client_model.decom_model(current_rank_ratio)
            client_tb = TensorBuffer(list(client_model.state_dict().values()))
            
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )   # 这里并不会阻塞，相当于告诉客户端数据放在哪里
            reqs.append(req)

        for req in reqs:
            req.wait()  # 这里会阻塞等待数据

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    def _fedavg(self, flatten_local_models, weights=None):
        n_selected_clients = len(flatten_local_models)

        if weights == None:
            weights = [
                torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
            ]
        # 存储所有客户端恢复后的全秩 StateDict
        recovered_state_dicts = []

        # NOTE: the arch for different local models needs to be the same as the master model.
        # retrieve the local models.
        local_models = {}
        for client_idx, flatten_local_model in flatten_local_models.items():
            _arch = self.clientid2arch[client_idx]
            # _model = copy.deepcopy(self.client_models[_arch])
             # _model = copy.deepcopy(_model)
            _, client_model = self.client_models[client_idx] # 1. 获取模型实例
           
            # 2. [确保状态] 必须处于分解状态才能 unpack buffer
            current_rank_ratio = self.conf.rank_list[client_idx - 1]
            if hasattr(client_model, "decom_model"):
                client_model.decom_model(current_rank_ratio)
            
            # 3. Unpack 数据 (将 Buffer 填入模型的 U, V 参数中)
            current_state_dict = client_model.state_dict()
            flatten_local_model.unpack(current_state_dict.values())
            client_model.load_state_dict(current_state_dict)

            # 4. [关键] 恢复全秩 (Recover)
            if hasattr(client_model, "recover_model"):
                client_model.recover_model()
            # 5. 保存恢复后的全秩参数用于后续平均
            recovered_state_dicts.append(copy.deepcopy(client_model.state_dict()))
        # 6. 开始聚合 (FedAvg)
        # 取第一个模型作为累加基底
        avg_model_state = copy.deepcopy(recovered_state_dicts[0])
            # 将参数转换为 Tensor 进行加权
        for key in avg_model_state.keys():
            # 初始化为第一个模型的加权值
            if "num_batches_tracked" in key:
                 # BatchNorm 的计数参数通常取 sum 或者 max，这里简化处理保持第一个
                 avg_model_state[key] = avg_model_state[key]
            else:
                avg_model_state[key] = avg_model_state[key] * weights[0]
                
                # 累加后续模型
                for i in range(1, n_selected_clients):
                     avg_model_state[key] += recovered_state_dicts[i][key] * weights[i]
        
        # 返回聚合后的参数字典 (此时是全秩的)
        return avg_model_state

            # _model_state_dict = self.client_models[_arch].state_dict()
            # _, client_model = self.client_models[client_idx]
            # _model_state_dict = copy.deepcopy(client_model.state_dict())

            # flatten_local_model.unpack(_model_state_dict.values())
            # _model.load_state_dict(_model_state_dict)
            # local_models[client_idx] = _model

        # uniformly average the local models.
        # assume we use the runtime stat from the last model.
        # _model = copy.deepcopy(_model)
        # local_states = [
        #     ModuleState(copy.deepcopy(local_model.state_dict()))
        #     for _, local_model in local_models.items()
        # ]
        # model_state = local_states[0] * weights[0]
        # for idx in range(1, len(local_states)):
        #     model_state += local_states[idx] * weights[idx]
        # model_state.copy_to_module(_model)
        # return _model

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )

            fedavg_model = self._fedavg(flatten_local_models) # 平均算法
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def aggregate(self, flatten_local_models):
        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        return fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        # aggregate the local models.
        # client_models = self.aggregate(
        #     flatten_local_models
        # ) # 计算平均后的模型

        # 1. 调用修改后的 _fedavg 获取全秩的聚合参数
        aggregated_state_dict = self._fedavg(flatten_local_models)

        self.master_model.load_state_dict(
            aggregated_state_dict
        ) # 更新全局模型
        # for arch, _client_model in client_models.items():
        #     self.client_models[arch].load_state_dict(_client_model.state_dict())
        
        # for arch, _client_model in client_models.items():
        #     for client_id, (model_name, model_instance) in self.client_models.items():
        #          if self.clientid2arch[client_id] == arch:
        #             model_instance.load_state_dict(_client_model.state_dict())
        # evaluate the aggregated model on the test data.
        master_utils.do_validation( # 最终评估
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label=f"aggregated_test_loader_0",
        )

        if not self.conf.train_fast:  # test all the selected_clients
            for client_idx, flatten_local_model in flatten_local_models.items():
                _arch = self.clientid2arch[client_idx]
                _model_state_dict = copy.deepcopy(self.client_models[_arch].state_dict())
                flatten_local_model.unpack(_model_state_dict.values())
                _, test_model = create_model.define_model(self.conf, to_consistent_model=False, arch=_arch)
                test_model.load_state_dict(_model_state_dict)

                master_utils.do_validation(
                    conf=self.conf,
                    coordinator=self.local_coordinator[client_idx - 1],
                    model=test_model,
                    criterion=self.criterion,
                    metrics=self.metrics,
                    data_loaders=self.test_loaders,
                    label=f"aggregated_test_loader_{client_idx}",
                )
            self.additional_validation()


        torch.cuda.empty_cache()
    def additional_validation(self):
        pass

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:   # 是否设置目标性能
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
