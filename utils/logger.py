"""
日志记录工具
用于将控制台输出同时保存到文件
"""

import logging
import os
import sys
from datetime import datetime


class Logger:
    """双输出日志类：同时输出到控制台和文件"""
    
    def __init__(self, log_file=None, log_dir='./logs', name='federated_learning'):
        """
        初始化logger
        
        Args:
            log_file: 日志文件路径，如果为None则自动生成
            log_dir: 日志目录
            name: logger名称
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # 清除已有的handlers
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 如果没有指定日志文件，则自动生成
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        else:
            log_file = os.path.join(log_dir, log_file)
        
        self.log_file = log_file
        
        # 创建formatter
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def info(self, message):
        """记录info级别日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录warning级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录error级别日志"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录debug级别日志"""
        self.logger.debug(message)
    
    def print_and_log(self, message):
        """
        打印并记录（兼容原有的print用法）
        这个方法不添加日志前缀，直接输出原始信息
        """
        # 直接写入文件（不带日志格式）
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(str(message) + '\n')
        # 输出到控制台
        print(message)
    
    def separator(self, char='=', length=80):
        """打印分隔线"""
        self.print_and_log(char * length)
    
    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file


class DualOutput:
    """
    双输出类：可以替代sys.stdout，将print同时输出到文件和控制台
    """
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logger(log_dir='./logs', experiment_name=None):
    """
    快速设置logger的便捷函数
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称，用于生成日志文件名
    
    Returns:
        Logger实例
    """
    if experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{experiment_name}_{timestamp}.log'
    else:
        log_file = None
    
    return Logger(log_file=log_file, log_dir=log_dir)


def redirect_stdout_to_file(log_file):
    """
    重定向stdout到文件（同时保持控制台输出）
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        DualOutput实例
    """
    dual_output = DualOutput(log_file)
    sys.stdout = dual_output
    return dual_output
