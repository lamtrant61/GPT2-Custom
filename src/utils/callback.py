import os
from transformers import TrainerCallback

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold=None, logger=None):
        self.threshold = threshold
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Ensure logs is not None
        if logs is None:
            return

        if (self.threshold is None) or (self.threshold <= 0):
                return

        # Check if 'loss' is in logs
        if 'loss' in logs:
            current_loss = logs['loss']

            # Stop training if loss is below or equal to the threshold
            if current_loss <= self.threshold:
                print(f"\nStopping training as loss has reached the threshold: {current_loss}")
                control.should_training_stop = True
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.logger is None:
            return
            
        # Lấy giá trị loss từ trainer
        logs = state.log_history[-1]  # Lấy logs của epoch cuối cùng

        if 'loss' in logs:
            self.logger.info(f"Epoch {state.epoch}, Loss: {logs['loss']}")

class CustomSaveModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer, epoch_save=10, path='./models'):
        self.epoch_save = epoch_save
        self.model = model
        self.tokenizer = tokenizer
        self.path = path
    
    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = state.epoch

        if (current_epoch % self.epoch_save == 0) and (current_epoch > 0):
            directory = os.path.abspath(self.path)
            if os.path.exists(directory):
                shutil.rmtree(directory)

            self.model.save_pretrained(self.path)
            self.tokenizer.save_pretrained(self.path)
            print(f"\nModel saved at epoch {current_epoch}")
