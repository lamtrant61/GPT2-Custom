from transformers import TrainerCallback

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold, logger=None):
        self.threshold = threshold
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Ensure logs is not None
        if logs is None:
            return

        # Check if 'loss' is in logs
        if 'loss' in logs:
            current_loss = logs['loss']
            # print(f"\nCurrent loss: {current_loss}")

            # Stop training if loss is below or equal to the threshold
            if current_loss <= self.threshold:
                print(f"\nStopping training as loss has reached the threshold: {current_loss}")
                control.should_training_stop = True
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # Lấy giá trị loss từ trainer
        logs = state.log_history[-1]  # Lấy logs của epoch cuối cùng
        # print ("log......................")
        # print (state.log_history[-1])
        if 'loss' in logs:
            self.logger.info(f"Epoch {state.epoch}, Loss: {logs['loss']}")
