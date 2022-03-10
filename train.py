from data.digits import Digits

def main():
    ### ------ Loading Data ------ ###
    train_set = Digits(train=True)
    test_set = Digits(train=False)
    breakpoint()


if __name__ == '__main__':
    main()
