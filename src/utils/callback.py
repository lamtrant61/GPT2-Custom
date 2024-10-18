from transformers import TrainerCallback

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Kiểm tra nếu loss có trong logs
        if 'loss' in logs and logs['loss'] <= self.threshold:
            print(f"Stopping training as loss has reached the threshold: {logs['loss']}")
            control.should_training_stop = True