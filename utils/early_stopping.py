"""
Early Stopping Utility for Handwritten Digit Recognition CNN
=============================================================

This module implements early stopping, a regularization technique used to prevent
overfitting during neural network training. Overfitting occurs when a model learns
the training data too well, including its noise and peculiarities, leading to poor
generalization on unseen data.

What is Early Stopping?
-----------------------
Early stopping monitors a validation metric (such as accuracy or loss) during
training and halts the process when the metric stops improving. Instead of training
for a fixed number of epochs, the model stops once it is clear that further training
would not yield better results on the validation set.

Why is Early Stopping Important?
--------------------------------
1. Prevents overfitting: Stops training before the model memorizes training data.
2. Saves computational resources: Avoids unnecessary training epochs.
3. Acts as implicit regularization: Limits model complexity without modifying
   the loss function or architecture.
4. Preserves the best model state: By tracking the best epoch, you can restore
   the model weights from the point of peak performance.

Key Parameters:
---------------
- patience: The number of consecutive epochs without improvement to tolerate
  before stopping. A higher patience allows the model more time to recover from
  temporary plateaus but risks overfitting if set too high.

- min_delta: The minimum change in the monitored metric that qualifies as an
  improvement. This prevents the training from stopping due to negligible
  fluctuations. For example, a min_delta of 0.001 means that an improvement
  of less than 0.1% is not considered meaningful.

Example Usage in a Training Loop:
----------------------------------
    from utils.early_stopping import EarlyStopping

    # Create early stopping monitor for validation accuracy (higher is better)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max', verbose=True)

    for epoch in range(num_epochs):
        # Train the model
        train_loss = train_one_epoch(model, train_loader, optimizer)

        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader)

        # Check early stopping condition
        if early_stopping(val_accuracy, epoch):
            print(f"Training stopped early at epoch {epoch}")
            print(f"Best accuracy: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
            break

    # Optionally retrieve the status
    status = early_stopping.get_status()
    print(f"Final status: {status}")
"""


class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric stops improving.

    This class tracks the best value of a given metric across epochs and counts
    the number of consecutive epochs without improvement. Once the patience
    threshold is exceeded, it signals that training should stop.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): 'max' if higher metric values are better (e.g., accuracy),
                    'min' if lower metric values are better (e.g., loss).
        verbose (bool): Whether to print status messages during training.
        best_score (float or None): The best metric value observed so far.
        counter (int): Number of consecutive epochs without improvement.
        should_stop (bool): Flag indicating whether training should be stopped.
        best_epoch (int): The epoch at which the best score was observed.
    """

    def __init__(self, patience=5, min_delta=0.001, mode='max', verbose=True):
        """
        Initialize the EarlyStopping monitor.

        Args:
            patience (int): Number of epochs to wait for improvement before
                stopping training. For example, patience=5 means training will
                continue for 5 more epochs after the last improvement before
                stopping. Default: 5.
            min_delta (float): Minimum change in the monitored metric to qualify
                as an improvement. This helps ignore trivial improvements that
                may be due to noise rather than genuine learning. Default: 0.001.
            mode (str): One of 'max' or 'min'.
                - 'max': The metric is expected to increase (e.g., accuracy).
                  An improvement means current_value > best_score + min_delta.
                - 'min': The metric is expected to decrease (e.g., loss).
                  An improvement means current_value < best_score - min_delta.
                Default: 'max'.
            verbose (bool): If True, print messages about early stopping status
                including improvements, warnings about no improvement, and stop
                notifications. Default: True.

        Raises:
            ValueError: If mode is not 'max' or 'min'.
        """
        if mode not in ('max', 'min'):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # Initialize tracking variables
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, current_value, epoch):
        """
        Evaluate whether training should stop based on the current metric value.

        This method compares the current metric value against the best observed
        value (accounting for min_delta). If there is an improvement, the best
        score and epoch are updated and the counter is reset. If there is no
        improvement, the counter is incremented toward the patience limit.

        Args:
            current_value (float): The current value of the monitored metric
                (e.g., validation accuracy or validation loss).
            epoch (int): The current epoch number (used for tracking and messages).

        Returns:
            bool: True if training should stop (patience exceeded), False otherwise.
        """
        # First call: initialize best_score with the current value
        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch
            if self.verbose:
                print(
                    f"[EarlyStopping] Initialized. "
                    f"Baseline score: {current_value:.6f} at epoch {epoch}"
                )
            return False

        # Determine if the current value is an improvement
        if self._is_improvement(current_value):
            # Improvement detected: update best score and reset counter
            if self.verbose:
                print(
                    f"[EarlyStopping] Improvement found at epoch {epoch}. "
                    f"Score: {current_value:.6f} (previous best: {self.best_score:.6f})"
                )
            self.best_score = current_value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement: increment counter
            self.counter += 1

            if self.counter >= self.patience:
                # Patience exceeded: signal to stop training
                self.should_stop = True
                if self.verbose:
                    print(
                        f"[EarlyStopping] Stopping training at epoch {epoch}. "
                        f"No improvement for {self.counter} consecutive epochs. "
                        f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}"
                    )
                return True
            else:
                # Patience not yet exceeded: warn and continue
                if self.verbose:
                    print(
                        f"[EarlyStopping] No improvement at epoch {epoch}. "
                        f"Counter: {self.counter}/{self.patience}. "
                        f"Current: {current_value:.6f}, Best: {self.best_score:.6f}"
                    )
                return False

    def _is_improvement(self, current_value):
        """
        Check whether the current value represents an improvement over the best score.

        For 'max' mode, an improvement means the current value exceeds the best
        score by at least min_delta. For 'min' mode, an improvement means the
        current value is below the best score by at least min_delta.

        Args:
            current_value (float): The current metric value to evaluate.

        Returns:
            bool: True if the current value is an improvement, False otherwise.
        """
        if self.mode == 'max':
            # Higher is better: improvement if current > best + min_delta
            return current_value > self.best_score + self.min_delta
        else:
            # Lower is better: improvement if current < best - min_delta
            return current_value < self.best_score - self.min_delta

    def reset(self):
        """
        Reset all tracking variables to their initial state.

        This is useful when you want to reuse the same EarlyStopping instance
        for a new training run without creating a new object. All counters,
        best scores, and flags are cleared.
        """
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def get_status(self):
        """
        Return a dictionary with the current state of the early stopping monitor.

        This is useful for logging, checkpointing, or debugging the training
        process. It provides a snapshot of all relevant tracking variables.

        Returns:
            dict: A dictionary containing:
                - best_score (float or None): The best metric value observed.
                - counter (int): Current count of epochs without improvement.
                - patience (int): Maximum allowed epochs without improvement.
                - best_epoch (int): The epoch at which the best score was observed.
                - should_stop (bool): Whether training should stop.
        """
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'patience': self.patience,
            'best_epoch': self.best_epoch,
            'should_stop': self.should_stop,
        }


def create_early_stopping(patience=5, min_delta=0.001, mode='max', verbose=True):
    """
    Convenience function to create an EarlyStopping instance.

    This function provides a simple factory interface for creating early stopping
    monitors without directly importing and instantiating the class. It is
    particularly useful in configuration-driven training pipelines where the
    early stopping parameters might be loaded from a config file or command-line
    arguments.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
            Default: 5.
        min_delta (float): Minimum change to qualify as an improvement.
            Default: 0.001.
        mode (str): 'max' for metrics where higher is better (e.g., accuracy),
            'min' for metrics where lower is better (e.g., loss).
            Default: 'max'.
        verbose (bool): Whether to print early stopping status messages.
            Default: True.

    Returns:
        EarlyStopping: A configured EarlyStopping instance ready to be used
            in a training loop.

    Example:
        >>> early_stopping = create_early_stopping(patience=10, mode='min')
        >>> # Use in training loop to monitor validation loss
        >>> for epoch in range(100):
        ...     val_loss = train_and_evaluate(model)
        ...     if early_stopping(val_loss, epoch):
        ...         break
    """
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        verbose=verbose,
    )
