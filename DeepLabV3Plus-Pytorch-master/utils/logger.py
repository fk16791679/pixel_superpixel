import os
import json
import logging
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='logs', experiment_name=None):
        # 创建日志目录
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 设置实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name
        
        # 设置日志文件路径
        self.log_file = os.path.join(log_dir, f'{experiment_name}.log')
        
        # 记录训练配置的文件路径，与日志文件保存在同一目录
        self.config_file = os.path.join(log_dir, f'{experiment_name}_config.json')
        
        # 配置日志记录器
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config):
        """记录训练配置"""
        # 保存配置到JSON文件
        with open(self.config_file, 'w') as f:
            json.dump(vars(config), f, indent=4)
        
        # 记录配置信息到日志
        self.logger.info('Training Configuration:')
        for key, value in vars(config).items():
            self.logger.info(f'{key}: {value}')
    
    def log_epoch(self, epoch, losses, metrics):
        """记录每个epoch的训练信息"""
        self.logger.info(f'\nEpoch {epoch}:')
        
        # 记录损失
        self.logger.info('Losses:')
        for loss_name, loss_value in losses.items():
            self.logger.info(f'  {loss_name}: {loss_value:.4f}')
        
        # 记录评估指标
        if metrics:
            self.logger.info('Metrics:')
            for metric_name, metric_value in metrics.items():
                self.logger.info(f'  {metric_name}: {metric_value:.4f}')
    
    def log_iteration(self, epoch, iteration, total_iterations, losses):
        """记录每次迭代的训练信息"""
        loss_str = ', '.join([f'{name}: {value:.4f}' for name, value in losses.items()])
        self.logger.info(
            f'Epoch {epoch}, Iteration {iteration}/{total_iterations} - {loss_str}'
        )
    
    def log_validation(self, epoch, metrics):
        """记录验证结果"""
        self.logger.info(f'\nValidation Results (Epoch {epoch}):')
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f'  {metric_name}: {metric_value:.4f}')
            else:
                self.logger.info(f'  {metric_name}: {metric_value}')
    
    def log_best_model(self, epoch, score):
        """记录最佳模型信息"""
        self.logger.info(f'\nNew Best Model at Epoch {epoch}:')
        self.logger.info(f'Score: {score:.4f}')
    
    def log_training_completed(self):
        """记录训练完成信息"""
        self.logger.info('\nTraining completed!')