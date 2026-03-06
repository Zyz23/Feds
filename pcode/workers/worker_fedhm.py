# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pcode.local_training.random_reinit as random_reinit
import pcode.datasets.mixup_data as mixup
import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
import  time
import copy
try:
    import clip
except ImportError:
    print("Warning: CLIP not installed. Please install via 'pip install git+https://github.com/openai/CLIP.git'")


class WorkerFedHM(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=conf.logger.log_metric,
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
        )

        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # 新增字典，缓存每一个客户端的clip映射层的权重
        self.local_adapters = {}

        # 初始化clip模型init_clip_model 加载一次
        self.init_clip_model()

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )   # 打印当前的进程编号

    # CLIP 初始化函数
    def init_clip_model(self):
        self.clip_device = self.device
        # 加载 ViT-B/32 模型，可以根据显存情况更换
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.clip_device)
        self.clip_model.eval()
        # 冻结 CLIP 参数，确保不参与更新也不上传
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 定义数据集类别名称
        if self.conf.data == 'cifar10':
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.conf.data == 'cifar100':
            # 这里简化处理，实际需填入 cifar100 的 100 个类名
            # 如果没有 metadata，可以使用数字代替，虽然语义效果会打折
            self.class_names = [f"class {i}" for i in range(100)]
        else:
            self.class_names = [f"class {i}" for i in range(self.conf.num_classes)]

        print("Generating CLIP prototypes for all classes...")
        prompts = [f"a photo of a {name}" for name in self.class_names]
        text_inputs = clip.tokenize(prompts).to(self.clip_device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
        
        # 归一化原型
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 存储所有原型，维度: [num_classes, clip_dim]   
        self.all_prototypes = text_features.float() # 保持所有原型以便索引，但只有 local classes 会被用到

    # # [ADD] 获取本地数据拥有的类别并生成原型
    # def generate_local_prototypes(self):
    #     # 创建一个临时的 loader 来扫描当前 client 拥有的类别
    #     temp_loader, _ = create_dataset.define_data_loader(
    #         self.conf,
    #         dataset=self.dataset["train"],
    #         localdata_id=self.conf.graph.client_id - 1,
    #         is_train=True,
    #         data_partitioner=self.data_partitioner,
    #     )
        
    #     # 统计本地拥有的类别索引
    #     local_classes = set()
    #     # 只需要遍历一小部分或者全部，取决于 partition 方式，为了准确通常遍历全部
    #     # 如果数据量大，可以考虑只取前几个 batch (假设 shuffle 足够好)
    #     for _, target in temp_loader:
    #         local_classes.update(target.tolist())
        
    #     self.local_classes_list = list(local_classes)
    #     # print(f"Client {self.conf.graph.client_id} local classes: {self.local_classes_list}")

    #     # 为这些类别生成原型
    #     prompts = [f"a photo of a {self.class_names[i]}" for i in range(len(self.class_names))]
    #     text_inputs = clip.tokenize(prompts).to(self.clip_device)

    #     with torch.no_grad():
    #         text_features = self.clip_model.encode_text(text_inputs)
        
    #     # 归一化原型
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
    #     # 存储原型：使用一个 buffer，索引对应 label
    #     # 维度: [num_classes, clip_dim]
    #     self.all_prototypes = text_features.float() # 保持所有原型以便索引，但只有 local classes 会被用到

    def recv_extra_info_from_master(self):
        pass

    def send_extra_info_to_master(self):
        pass

    def get_params(self, model):
        weight_collector = None
        for param in model.parameters():
            if not isinstance(weight_collector, torch.Tensor):
                weight_collector = param.reshape(-1)
            else:
                weight_collector = torch.cat((weight_collector, param.reshape(-1)), 0)
        return weight_collector

    def run(self):
        while True:
            self.listen_to_master() # 每次可能分配不同的客户端

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self.recv_extra_info_from_master()

            self._recv_model_from_master() # 接受模型参数
            self._train()

            self.send_extra_info_to_master()

            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return
    def extra_init(self):
        # # 1. 生成当前 Client 的本地类别原型
        # self.generate_local_prototypes()

        # 2. 检查维度并添加适配层
        # 我们需要运行一次 dummy forward 来确定 feature 的维度

        # [FIX] 确保模型在执行 dummy forward 之前已经在 GPU 上
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)

        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, 3, 32, 32).to(self.device) # 假设输入是 32x32
        #     feature, _ = self.model(dummy_input)
        #     model_feat_dim = feature.shape[1]
        #     clip_feat_dim = self.all_prototypes.shape[1]

        # # 添加适配层
        # # 将适配层注册为 self.model 的子模块，这样它会自动进入 optimizer
        # self.model.clip_adapter = nn.Linear(model_feat_dim, clip_feat_dim).to(self.device)
        # # 获取当前正在模拟的 Client ID
        # client_id = self.conf.graph.client_id    
        # # [ADD] 检查本地字典中是否已经存有该 Client 的历史 adapter 权重
        # if client_id in self.local_adapters:
        #     # 如果有，加载历史权重，接着上一轮继续训练（修复灾难性遗忘）
        #     self.model.clip_adapter.load_state_dict(self.local_adapters[client_id])
        #     # print(f"Client-{client_id} loaded historical CLIP adapter weights.")
        # else:
        #     # 如果是该 Client 第一次参与训练，进行随机初始化
        #     nn.init.xavier_normal_(self.model.clip_adapter.weight)
        #     nn.init.zeros_(self.model.clip_adapter.bias) # bias也最好清零
        #     # print(f"Client-{client_id} initialized a new CLIP adapter.")

    def listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        ) # 取出属于自己的消息

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        ) # 获取模型

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values())) # 将参数打包为一维参数，方便传输

        self.extra_init()

        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=self.conf.graph.client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )   # 这里的data_partitioner已经统一,这里传入的参时真正的客户端id

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values()) # 将接受的参数填入模型字典

        # if not self.conf.freeze_bn:
        self.model.load_state_dict(self.model_state_dict,strict=False)   # 更新模型参数 strict=False 以容忍 adapter 参数缺失
        random_reinit.random_reinit_model(self.conf, self.model)    # 若为true，这里可能会随机丢弃参数
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device)) # 获得一个关闭了梯度的一摸一样的模型，常用于蒸馏

        # self.aggregation = Aggregation(self.model.classifier.in_features).cuda()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )

        dist.barrier()

    def _prepare_train(self):
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)

# 把 clip_adapter 挂在 self.model 下，create_optimizer 会自动识别到它的参数
        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )   # 建立优化器，同时加入L2正则
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names) # 创建追踪器，记录损失和准确率
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )

    def prepare_train(self):
        self._prepare_train()

    # 添加正则项 和使用原型的mse损失    
    def local_training_with_extra_calculate(self, loss, output, data_batch, feature):
        if hasattr(self.model, "get_decomposed_loss"):
            # 2. 计算正则损失
            # type 可选: 'frobenius', 'kronecker', 'l2' 
            decom_reg = self.model.get_decomposed_loss(type='frobenius') 
            
            # 3. 获取正则化系数 (Lambda)
            # 在 conf 中定义 decom_lambda，
            # 如果系数太大，会导致模型不关注分类准确率；太小则无法保持低秩特性
            
            # 4. 将正则项加到总 Loss 中
            loss += self.conf.meta_L2 * decom_reg

        # --- CLIP Prototype Alignment Loss ---
        if hasattr(self, "all_prototypes"):
            # 1. 映射特征到 CLIP 空间 (如果需要)
            # 确保使用 self.model.clip_adapter
            if hasattr(self.model, 'clip_adapter'):
                aligned_feature = self.model.clip_adapter(feature)
            else:
                aligned_feature = feature
            
            # 2. 归一化特征 (CLIP 使用 cosine similarity，通常做 L2 norm)
            # 你的 已经对 feature 做了 normalize，但经过 adapter 后需要重新 normalize
            aligned_feature = F.normalize(aligned_feature, p=2,dim=1)

            # 3. 获取当前 batch 对应的原型
            # data_batch["target"] 包含了 label
            # self.all_prototypes 是 [num_classes, dim]

            targets = data_batch["target"]
            target_prototypes = self.all_prototypes[targets] # [batch_size, dim]

            # 4. 计算 MSE Loss
            mse_loss = F.mse_loss(aligned_feature, target_prototypes)

            # 5. 加权添加到主 loss (权重可以设为超参数，这里暂时设为 1.0 或 0.5)
            # 这里默认 1.0，可以根据需要调整
            loss +=  2.0*mse_loss

            # 3. [MODIFIED] 计算与所有类别原型的余弦相似度矩阵
            # aligned_feature 维度: [batch_size, clip_dim]
            # all_prototypes 维度: [num_classes, clip_dim]
            # 矩阵乘法后 logits 维度: [batch_size, num_classes]
            # 因为两者都经过了 L2 归一化，点乘的结果即为余弦相似度 (-1 到 1)
            # logits = torch.matmul(aligned_feature, self.all_prototypes.T)

            # 4. [MODIFIED] 引入温度系数 (Temperature scaling)
            # 对比学习中余弦相似度必须除以温度系数来放大差异，否则 Softmax 算出的梯度会极度平缓
            # 0.07 是 SimCLR/MoCo/CLIP 中极其经典且管用的默认超参数 (1/0.07 ≈ 14.28)
            # tau = 0.5
            # logits = logits / tau

            # 5. [MODIFIED] 使用交叉熵损失计算对齐 Loss
            # 这会自动把特征拉向正确的文本原型，并推开其他所有错误的文本原型
            # targets = data_batch["target"]
            # align_loss = F.cross_entropy(logits, targets)

            # 6. 加权添加到主 loss 
            # 经过温度放大后的 align_loss 通常在数值量级上与主任务的 CrossEntropyLoss 相当
            # 这里的权重 0.5 就可以起到很好的平衡作用 (视实验结果可微调为 0.1 ~ 1.0)
            # loss += 0.1 * align_loss

        return loss 

    def add_grad(self):
        pass

    def _train(self):
        self.prepare_train() # 初始化指标追踪器，优化器等
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:

            for _input, _target in self.train_loader:

                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_): # 数据统计
                    data_batch = create_dataset.load_data_batch(    # 这里主要是将数据搬到cuda上
                        self.conf, _input, _target, is_training=True, device=self.device
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    loss, performance, feature, output = self._inference(data_batch)    # 前向传播 计算损失

                with self.timer("extra_forward_pass", epoch=self.scheduler.epoch_):
                    # 添加额外的计算 分解的损失
                    loss = self.local_training_with_extra_calculate(loss, output, data_batch, feature)  # 这里没有额外的训练

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10) # 梯度裁剪，max_norm为允许的最大梯度

                    self.optimizer.step()
                    self.scheduler.step()

                # check divergence.检查loss是否过大，防止将有毒参数上传
                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )   # 报告哪个客户端出错

                    self._terminate_comm_round()    # 终止训练
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round(): # 检查轮数，如果够20轮直接返回

                    self.conf.logger.log(self.timer.summary())

                    self._terminate_comm_round()
                    return

            # # display tracking time.
            # if (
            #         self.conf.display_tracked_time
            #         and self.scheduler.local_index % self.conf.summary_freq == 0
            # ):


            # display the logging info.
            display_training_stat(self.conf, self.scheduler, self.tracker) # 打印这一轮的成功
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        feature, output = self.model(data_batch["input"]) 

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            bsz = data_batch["target"].size(0)
            self.tracker.update_local_metrics(
                loss.item(), 0, n_samples=bsz
            )
            for idx in range(1, 1 + len(performance)):
                self.tracker.update_local_metrics(
                    performance[idx - 1], idx, n_samples=bsz
                )
        return loss, performance, feature, output

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )

        
        # [MODIFIED] 
        # 因为我们是把 clip_adapter 挂在 model 下的，state_dict 会包含它
        # 必须过滤掉
        state_dict = self.model.state_dict()
        # to_send_dict = {k: v for k, v in state_dict.items() if 'clip_adapter' not in k}

        # flatten_model = TensorBuffer(list(to_send_dict.values()))    # 打包数据
        # 这样天然过滤掉了不属于原始模型结构的 clip_adapter，同时保证顺序 100% 正确
        safe_send_values = [state_dict[k] for k in self.model_state_dict.keys()]

        flatten_model = TensorBuffer(safe_send_values)    # 打包数据
        dist.send(tensor=flatten_model.buffer, dst=0)   # 发送
        dist.barrier()      # 等待其他客户端

    def _terminate_comm_round(self):
        self.model = self.model.cpu()

        self.scheduler.clean()
        self.conf.logger.save_json()
        torch.cuda.empty_cache()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False


