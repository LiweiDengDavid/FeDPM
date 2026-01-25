import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
import copy

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id, dataset_name, args, device, logger=None):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.args = copy.deepcopy(args)
        self.device = device
        self.logger = logger
        
        # 为当前客户端设置数据集特定参数
        self._setup_dataset_config()
        
        # 获取数据加载器
        self.train_data, self.train_loader = data_provider(self.args, 'train')
        self.vali_data, self.vali_loader = data_provider(self.args, 'val')
        self.test_data, self.test_loader = data_provider(self.args, 'test')
        
        # 获取训练样本数量（用于FedAvg加权）
        self.num_samples = len(self.train_data)
        
        # 初始化模型（只初始化一次，避免每轮重新初始化）
        from models import TimeMixer
        self.model = None  # 稍后在第一次训练时初始化
        
        msg = (f"Client {self.client_id} ({self.dataset_name}): "
               f"Train samples: {self.num_samples}, "
               f"Channels: {self.args.enc_in}")
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
    
    def _setup_dataset_config(self):
        """为不同数据集设置特定配置"""
        dataset_configs = {
            'ETTh1': {
                'data': 'ETTh1',
                'root_path': './dataset/ETT-small/',
                'data_path': 'ETTh1.csv',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'freq': 'h'
            },
            'ETTh2': {
                'data': 'ETTh2',
                'root_path': './dataset/ETT-small/',
                'data_path': 'ETTh2.csv',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'freq': 'h'
            },
            'ETTm1': {
                'data': 'ETTm1',
                'root_path': './dataset/ETT-small/',
                'data_path': 'ETTm1.csv',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'freq': 't'
            },
            'ETTm2': {
                'data': 'ETTm2',
                'root_path': './dataset/ETT-small/',
                'data_path': 'ETTm2.csv',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7,
                'freq': 't'
            },
            'weather': {
                'data': 'custom',
                'root_path': './dataset/weather/',
                'data_path': 'weather.csv',
                'enc_in': 21,
                'dec_in': 21,
                'c_out': 21,
                'freq': 'h'
            },
            'electricity': {
                'data': 'custom',
                'root_path': './dataset/electricity/',
                'data_path': 'electricity.csv',
                'enc_in': 321,
                'dec_in': 321,
                'c_out': 321,
                'freq': 'h'
            },
            'exchange': {
                'data': 'custom',
                'root_path': './dataset/exchange_rate/',
                'data_path': 'exchange_rate.csv',
                'enc_in': 8,
                'dec_in': 8,
                'c_out': 8,
                'freq': 'h'
            }
        }
        
        if self.dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        config = dataset_configs[self.dataset_name]
        for key, value in config.items():
            setattr(self.args, key, value)
    
    def get_model(self, model_class):
        """创建客户端模型"""
        model = model_class(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)
    
    def local_train(self, global_shared_params, local_epochs):
        """本地训练"""
        # 第一次训练时初始化模型，之后复用同一个模型
        if self.model is None:
            from models import TimeMixer
            self.model = self.get_model(TimeMixer.Model)
            msg = f"Client {self.client_id}: Model initialized for the first time"
            if self.logger:
                self.logger.print_and_log(msg)
            else:
                print(msg)
        
        # 加载全局共享参数
        if global_shared_params is not None:
            self.set_shared_params(global_shared_params)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        
        # 使用StepLR替代OneCycleLR，更适合联邦学习的短期训练
        scheduler = lr_scheduler.StepLR(
            optimizer=model_optim,
            step_size=max(1, local_epochs // 2),  # 在中期降低学习率
            gamma=0.5
        )
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        msg = f"\nClient {self.client_id} ({self.dataset_name}) - Local Training"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        for epoch in range(local_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None
                
                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                # Backward
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                # 每个epoch结束后更新学习率
            
            scheduler.step()  # StepLR在每个epoch后调用
            
            train_loss = np.average(train_loss)
            msg = f"  Epoch {epoch + 1}/{local_epochs} - Loss: {train_loss:.7f} - Time: {time.time() - epoch_time:.2f}s"
            if self.logger:
                self.logger.print_and_log(msg)
            else:
                print(msg)
        
        # 返回共享参数
        return self.get_shared_params()
    
    def get_shared_params(self):
        """获取需要聚合的共享参数（只共享结构一致的层）"""
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        shared_params = OrderedDict()
        state_dict = model.state_dict()
        
        # 只共享PDM blocks - 这是唯一在所有客户端都有完全相同结构的核心组件
        # 其他层因为以下原因无法共享：
        # - enc_embedding.temporal_embedding: 依赖于freq参数，不同数据集freq不同（'h' vs 't'）
        # - normalize_layers: 数量取决于down_sampling_layers + 1
        # - predict_layers: 维度取决于seq_len和down_sampling_window
        # - projection_layer: 输出维度取决于c_out
        
        for name, param in state_dict.items():
            if 'pdm_blocks' in name:
                shared_params[name] = param.cpu().clone()
        
        return shared_params
    
    def set_shared_params(self, shared_params):
        """设置共享参数到本地模型"""
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        state_dict = model.state_dict()
        
        # 更新共享参数
        updated_count = 0
        for name, param in shared_params.items():
            if name in state_dict:
                state_dict[name] = param.to(self.device)
                updated_count += 1
        
        model.load_state_dict(state_dict)
        
        # 可选：打印更新信息（仅第一次或debug时）
        # msg = f"Client {self.client_id}: Updated {updated_count} parameters from global model"
        # if self.logger:
        #     self.logger.print_and_log(msg)
    
    def validate(self, criterion):
        """验证"""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                
                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None
                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 【修复】只计算预测部分的损失，与训练时保持一致
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        return np.average(total_loss)


class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, args, clients, logger=None):
        self.args = args
        self.clients = clients
        self.logger = logger
        self.global_shared_params = None
        self.best_global_params = None
        self.best_vali_loss = float('inf')
    
    def fedavg_aggregate(self, client_params_list, client_sample_nums):
        """FedAvg聚合算法"""
        total_samples = sum(client_sample_nums)
        global_params = OrderedDict()
        
        # 获取所有参数的key
        param_keys = client_params_list[0].keys()
        
        for key in param_keys:
            # 加权平均
            global_params[key] = sum(
                client_params[key] * (num_samples / total_samples)
                for client_params, num_samples in zip(client_params_list, client_sample_nums)
            )
        
        return global_params
    
    def train_one_round(self, round_idx, local_epochs):
        """训练一轮"""
        msg = f"\n{'='*80}\nRound {round_idx + 1}/{self.args.num_rounds}\n{'='*80}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        # 所有客户端本地训练
        client_params_list = []
        client_sample_nums = []
        
        for client in self.clients:
            # 本地训练
            shared_params = client.local_train(self.global_shared_params, local_epochs)
            client_params_list.append(shared_params)
            client_sample_nums.append(client.num_samples)
        
        # FedAvg聚合
        msg = f"\n{'*'*80}\nServer: Aggregating models using FedAvg...\nAggregation completed!\n{'*'*80}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        self.global_shared_params = self.fedavg_aggregate(client_params_list, client_sample_nums)
        
        # 验证全局模型
        avg_vali_loss = self.validate_global_model()
        
        # 保存最佳模型
        if avg_vali_loss < self.best_vali_loss:
            self.best_vali_loss = avg_vali_loss
            self.best_global_params = copy.deepcopy(self.global_shared_params)
            msg = f"✓ New best global model! Avg Vali Loss: {avg_vali_loss:.7f}"
            if self.logger:
                self.logger.print_and_log(msg)
            else:
                print(msg)
        
        return avg_vali_loss
    
    def validate_global_model(self):
        """在所有客户端的验证集上验证全局模型"""
        msg = f"\n{'='*80}\nServer: Validating global model on all clients...\n{'='*80}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        criterion = nn.MSELoss()
        all_vali_losses = []
        
        for client in self.clients:
            # 加载全局共享参数
            from models import TimeMixer
            client.model = client.get_model(TimeMixer.Model)
            client.set_shared_params(self.global_shared_params)
            
            # 验证
            vali_loss = client.validate(criterion)
            all_vali_losses.append(vali_loss)
            msg = f"  Client {client.client_id} ({client.dataset_name:12s}): Vali Loss = {vali_loss:.7f}"
            if self.logger:
                self.logger.print_and_log(msg)
            else:
                print(msg)
        
        avg_vali_loss = np.mean(all_vali_losses)
        msg = f"\n  Average Validation Loss: {avg_vali_loss:.7f}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        return avg_vali_loss
    
    def save_global_model(self, save_path):
        """保存全局模型（仅保存共享参数）"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        model_path = os.path.join(save_path, 'global_model.pth')
        torch.save(self.best_global_params, model_path)
        msg = f"\nGlobal model saved to: {model_path}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)


class Exp_Federated_Learning(Exp_Basic):
    """联邦学习实验类"""
    
    def __init__(self, args, logger=None):
        # 每个客户端会独立构建自己的模型
        self.args = args
        self.logger = logger
        self.device = self._acquire_device()
        self.clients = []
        self.server = None
        # model_dict 用于导入模型类
        from models import TimeMixer
        self.model_dict = {
            'TimeMixer': TimeMixer
        }
    
    def _acquire_device(self):
        """获取设备"""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            msg = 'Use GPU: cuda:{}'.format(self.args.gpu)
        else:
            device = torch.device('cpu')
            msg = 'Use CPU'
        
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        return device
    
    def _build_model(self):
        """构建模型（联邦学习中每个客户端独立构建）"""
        pass
    
    def _get_data(self, flag):
        """获取数据（联邦学习中每个客户端独立获取）"""
        pass
    
    def _select_optimizer(self):
        """选择优化器（联邦学习中每个客户端独立选择）"""
        pass
    
    def _select_criterion(self):
        """选择损失函数"""
        return nn.MSELoss()
    
    def _init_clients(self):
        """初始化所有客户端"""
        if self.logger:
            self.logger.separator('=', 80)
            self.logger.print_and_log("Initializing Federated Learning Clients...")
            self.logger.separator('=', 80)
        else:
            print("\n" + "="*80)
            print("Initializing Federated Learning Clients...")
            print("="*80)
        
        for idx, dataset_name in enumerate(self.args.client_datasets):
            client = FederatedClient(
                client_id=idx,
                dataset_name=dataset_name,
                args=self.args,
                device=self.device,
                logger=self.logger
            )
            self.clients.append(client)
        
        msg = f"\nTotal clients: {len(self.clients)}\nTotal training samples: {sum(c.num_samples for c in self.clients)}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
    
    def train(self, setting):
        """联邦学习训练主流程"""
        # 初始化客户端
        self._init_clients()
        
        # 初始化服务器
        self.server = FederatedServer(self.args, self.clients, logger=self.logger)
        
        # 创建保存路径
        save_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 联邦学习训练
        if self.logger:
            self.logger.separator('=', 80)
            self.logger.print_and_log("Starting Federated Learning Training...")
            self.logger.separator('=', 80)
            self.logger.print_and_log(f"Total Rounds: {self.args.num_rounds}")
            self.logger.print_and_log(f"Local Epochs per Round: {self.args.local_epochs}")
            self.logger.print_and_log(f"Channel Independence: {self.args.channel_independence}")
            self.logger.separator('=', 80)
        else:
            print("\n" + "="*80)
            print("Starting Federated Learning Training...")
            print("="*80)
            print(f"Total Rounds: {self.args.num_rounds}")
            print(f"Local Epochs per Round: {self.args.local_epochs}")
            print(f"Channel Independence: {self.args.channel_independence}")
            print("="*80)
        
        training_start_time = time.time()
        
        # Early stopping
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        for round_idx in range(self.args.num_rounds):
            round_start_time = time.time()
            
            # 训练一轮
            avg_vali_loss = self.server.train_one_round(round_idx, self.args.local_epochs)
            
            # Early stopping检查
            # early_stopping(avg_vali_loss, None, save_path)
            
            round_time = time.time() - round_start_time
            msg = f"\nRound {round_idx + 1} completed in {round_time:.2f}s\n{'='*80}\n"
            if self.logger:
                self.logger.print_and_log(msg)
            else:
                print(msg)
            
            # if early_stopping.early_stop:
            #     msg = "Early stopping triggered!"
            #     if self.logger:
            #         self.logger.print_and_log(msg)
            #     else:
            #         print(msg)
            #     break
        
        total_time = time.time() - training_start_time
        if self.logger:
            self.logger.separator('=', 80)
            self.logger.print_and_log("Federated Learning Training Completed!")
            self.logger.print_and_log(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            self.logger.print_and_log(f"Best Validation Loss: {self.server.best_vali_loss:.7f}")
            self.logger.separator('=', 80)
            self.logger.print_and_log("")
        else:
            print(f"\n{'='*80}")
            print(f"Federated Learning Training Completed!")
            print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"Best Validation Loss: {self.server.best_vali_loss:.7f}")
            print(f"{'='*80}\n")
        
        # 保存最佳全局模型
        self.server.save_global_model(save_path)
        
        return None
    
    def test(self, setting, test=0):
        """使用全局模型在所有客户端数据集上测试"""
        if self.logger:
            self.logger.separator('=', 80)
            self.logger.print_and_log("Testing Global Model on All Client Datasets...")
            self.logger.separator('=', 80)
        else:
            print("\n" + "="*80)
            print("Testing Global Model on All Client Datasets...")
            print("="*80)
        
        # 如果没有初始化客户端，则初始化
        if not self.clients:
            self._init_clients()
        
        # 加载全局模型
        model_path = os.path.join('./checkpoints/' + setting, 'global_model.pth')
        if not os.path.exists(model_path):
            msg = f"Error: Global model not found at {model_path}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return
        
        global_shared_params = torch.load(model_path)
        msg = f"Loaded global model from: {model_path}\n"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        # 创建结果保存路径
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 在每个客户端上测试
        all_results = {}
        
        for client in self.clients:
            if self.logger:
                self.logger.print_and_log(f"\n{'-'*80}")
                self.logger.print_and_log(f"Testing on Client {client.client_id}: {client.dataset_name}")
                self.logger.print_and_log(f"{'-'*80}")
            else:
                print(f"\n{'-'*80}")
                print(f"Testing on Client {client.client_id}: {client.dataset_name}")
                print(f"{'-'*80}")
            
            # 创建模型并加载全局参数
            from models import TimeMixer
            client.model = client.get_model(TimeMixer.Model)
            client.set_shared_params(global_shared_params)
            client.model.eval()
            
            # 测试
            preds = []
            trues = []
            
            # 创建客户端特定的结果文件夹
            client_folder = os.path.join(folder_path, client.dataset_name)
            if not os.path.exists(client_folder):
                os.makedirs(client_folder)
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(client.test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    if 'PEMS' == client.args.data or 'Solar' == client.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    
                    if client.args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -client.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :client.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        dec_inp = None
                    
                    # 推理
                    if client.args.output_attention:
                        outputs = client.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = client.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if client.args.features == 'MS' else 0
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    
                    pred = outputs
                    true = batch_y
                    
                    preds.append(pred)
                    trues.append(true)
                    
                    # 可视化（每20个batch保存一次）
                    if i % 20 == 0:
                        input_data = batch_x.detach().cpu().numpy()
                        if client.test_data.scale and client.args.inverse:
                            shape = input_data.shape
                            input_data = client.test_data.inverse_transform(input_data.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(client_folder, f'prediction_{i}.pdf'))
            
            # 合并所有预测
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            
            # 计算指标
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            
            if self.logger:
                self.logger.print_and_log(f"\n{client.dataset_name} Results:")
                self.logger.print_and_log(f"  MSE:  {mse:.7f}")
                self.logger.print_and_log(f"  MAE:  {mae:.7f}")
                self.logger.print_and_log(f"  RMSE: {rmse:.7f}")
                self.logger.print_and_log(f"  MAPE: {mape:.7f}")
                self.logger.print_and_log(f"  MSPE: {mspe:.7f}")
            else:
                print(f"\n{client.dataset_name} Results:")
                print(f"  MSE:  {mse:.7f}")
                print(f"  MAE:  {mae:.7f}")
                print(f"  RMSE: {rmse:.7f}")
                print(f"  MAPE: {mape:.7f}")
                print(f"  MSPE: {mspe:.7f}")
            
            # 保存结果
            all_results[client.dataset_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mspe': mspe
            }
            
            # 保存到文件
            result_file = os.path.join(client_folder, 'metrics.txt')
            with open(result_file, 'w') as f:
                f.write(f"Dataset: {client.dataset_name}\n")
                f.write(f"MSE:  {mse:.7f}\n")
                f.write(f"MAE:  {mae:.7f}\n")
                f.write(f"RMSE: {rmse:.7f}\n")
                f.write(f"MAPE: {mape:.7f}\n")
                f.write(f"MSPE: {mspe:.7f}\n")
        
        # 计算并保存平均结果
        if self.logger:
            self.logger.separator('=', 80)
            self.logger.print_and_log("Average Results Across All Clients:")
            self.logger.separator('=', 80)
        else:
            print(f"\n{'='*80}")
            print(f"Average Results Across All Clients:")
            print(f"{'='*80}")
        
        avg_results = {
            'mse': np.mean([r['mse'] for r in all_results.values()]),
            'mae': np.mean([r['mae'] for r in all_results.values()]),
            'rmse': np.mean([r['rmse'] for r in all_results.values()]),
            'mape': np.mean([r['mape'] for r in all_results.values()]),
            'mspe': np.mean([r['mspe'] for r in all_results.values()])
        }
        
        if self.logger:
            self.logger.print_and_log(f"Average MSE:  {avg_results['mse']:.7f}")
            self.logger.print_and_log(f"Average MAE:  {avg_results['mae']:.7f}")
            self.logger.print_and_log(f"Average RMSE: {avg_results['rmse']:.7f}")
            self.logger.print_and_log(f"Average MAPE: {avg_results['mape']:.7f}")
            self.logger.print_and_log(f"Average MSPE: {avg_results['mspe']:.7f}")
            self.logger.separator('=', 80)
            self.logger.print_and_log("")
        else:
            print(f"Average MSE:  {avg_results['mse']:.7f}")
            print(f"Average MAE:  {avg_results['mae']:.7f}")
            print(f"Average RMSE: {avg_results['rmse']:.7f}")
            print(f"Average MAPE: {avg_results['mape']:.7f}")
            print(f"Average MSPE: {avg_results['mspe']:.7f}")
            print(f"{'='*80}\n")
        
        # 保存汇总结果
        summary_file = os.path.join(folder_path, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Federated Learning Test Results Summary\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, results in all_results.items():
                f.write(f"{dataset_name}:\n")
                f.write(f"  MSE:  {results['mse']:.7f}\n")
                f.write(f"  MAE:  {results['mae']:.7f}\n")
                f.write(f"  RMSE: {results['rmse']:.7f}\n")
                f.write(f"  MAPE: {results['mape']:.7f}\n")
                f.write(f"  MSPE: {results['mspe']:.7f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Average Results:\n")
            f.write("="*80 + "\n")
            f.write(f"Average MSE:  {avg_results['mse']:.7f}\n")
            f.write(f"Average MAE:  {avg_results['mae']:.7f}\n")
            f.write(f"Average RMSE: {avg_results['rmse']:.7f}\n")
            f.write(f"Average MAPE: {avg_results['mape']:.7f}\n")
            f.write(f"Average MSPE: {avg_results['mspe']:.7f}\n")
        
        msg = f"Summary saved to: {summary_file}"
        if self.logger:
            self.logger.print_and_log(msg)
        else:
            print(msg)
        
        return
