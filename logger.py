import logging

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._init_logger()
        return cls._instance
    
    def _init_logger(self):
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        file_handler = logging.FileHandler("training.log")
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
    
    def log(self, message):
        self.logger.info(message)
