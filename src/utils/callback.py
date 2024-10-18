from transformers import TrainerCallback

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold

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