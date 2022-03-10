from data import utils as data_utils
from data.digits import Digits
from models.models import Model
from models import utils as model_utils

def main():
    ### ------ Loading Data ------ ###
    train_set = Digits(train=True)
    test_set = Digits(train=False)
    train_loader = data_utils.get_dataloader(train_set, batch_size=128)
    test_loader = data_utils.get_dataloader(test_set, batch_size=128)

    ### ------ Loading Model ------ ###
    model = Model(1, 64)

    ### ------ Loading Trainer ------ ###
    trainer = model_utils.Trainer(model, train_loader, 'Adam', 0.001)
    trainer.train(10)
    breakpoint()


if __name__ == '__main__':
    main()
