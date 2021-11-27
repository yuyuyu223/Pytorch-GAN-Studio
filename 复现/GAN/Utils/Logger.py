import logging
import os
import time


class logger(object):

    # 在这里定义StreamHandler，可以实现单例， 所有的logger()共用一个StreamHandler
    ch = logging.StreamHandler()

    def __init__(self, rank):
        self.logger = logging.getLogger()
        if not self.logger.handlers:
            # 如果self.logger没有handler， 就执行以下代码添加handler
            self.logger.setLevel(logging.INFO)
            self.log_path = "./log"
            # 创建一个handler,用于写入日志文件
            fh = logging.FileHandler(self.log_path + '/runlog-' + time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log', encoding='utf-8')
            # 只写入INFO级以上级别
            fh.setLevel(logging.INFO)

            # 定义handler的输出格式
            formatter = logging.Formatter(
                '[%(asctime)s] - [%(levelname)s] - %(message)s')
            fh.setFormatter(formatter)

            # 给logger添加handler
            self.logger.addHandler(fh)
            self.rank = rank

    def debug(self, message):
        if self.rank != 0:
            return
        self.fontColor('\033[0;32m%s\033[0m')
        self.logger.debug(message)

    def info(self, message):
        if self.rank != 0:
            return
        self.fontColor('\033[0;34m%s\033[0m')
        self.logger.info(message)

    def warning(self, message):
        if self.rank != 0:
            return
        self.fontColor('\033[0;37m%s\033[0m')
        self.logger.warning(message)

    def error(self, message):
        if self.rank != 0:
            return
        self.fontColor('\033[0;31m%s\033[0m')
        self.logger.error(message)

    def critical(self, message):
        if self.rank != 0:
            return
        self.fontColor('\033[0;35m%s\033[0m')
        self.logger.critical(message)

    def fontColor(self, color):
        
        # 不同的日志输出不同的颜色
        formatter = logging.Formatter(
            color % '[%(asctime)s] - [%(levelname)s] - %(message)s')
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.ch)


if __name__ == "__main__":
    logger = logger()
    logger.info("12345")
    logger.debug("12345")
    logger.warning("12345")
    logger.error("12345")
