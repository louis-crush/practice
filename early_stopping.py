import datetime
import torch

class EarlyStopping(object):
    def __init__(self, file_name_prefix, patience=10):
        dt = datetime.datetime.now()
        self.filename = file_name_prefix+"early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.best_micro_f1 = None
        self.best_macro_f1 = None
        self.early_stop = False
        self.best_epoch = None
        self.best_pred = None

    def step(self, loss, acc, micro_f1, macro_f1, epoch, raw_preds, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.best_micro_f1 = micro_f1
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch
            self.best_pred =raw_preds
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # todo 待修改，不确定if的条件是否合适
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
                # todo 这里应该要输出验证集相应的指标
            self.best_loss = min((loss, self.best_loss))
            self.best_acc = max((acc, self.best_acc))
            self.best_micro_f1 = max(micro_f1, self.best_micro_f1)
            self.best_macro_f1 = max(macro_f1, self.best_macro_f1)
            self.best_pred = raw_preds
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))