from data.digits import Digits
from models.models import Model
from data import utils as data_utils
from models import utils as model_utils

def main():
    ### ------ Loading Data ------ ###
    train_loader = data_utils.get_dataloader(Digits(train=True), batch_size=128)
    test_loader = data_utils.get_dataloader(Digits(train=False), batch_size=128)

    ### ------ Loading Model ------ ###
    model = Model(1, 64)

    ### ------ Loading Trainer ------ ###
    trainer = model_utils.Trainer(model, train_loader, 'Adam', 0.001)
    train_loss = trainer.train(10)
    test_acc = trainer.test(test_loader)
    breakpoint()


if __name__ == '__main__':
    main()
