dict(
    z = 20,
    batch_size = 16,
    label_size = 2,
    L = 1,
    learning_rate = 0.001,
    max_iter = 100,
    clip_c = 1.,
    is_load = False,
    seed = 1473769786,

    train_arg1 = './data/com/train.arg1.dta.pkl',
    train_arg2 = './data/com/train.arg2.dta.pkl',
    train_lbl = './data/com/train.lbl.dta.pkl',
    dev_arg1 = './data/com/dev.arg1.dta.pkl',
    dev_arg2 = './data/com/dev.arg2.dta.pkl',
    dev_lbl = './data/com/dev.lbl.dta.pkl',
    test_arg1 = './data/com/test.arg1.dta.pkl',
    test_arg2 = './data/com/test.arg2.dta.pkl',
    test_lbl = './data/com/test.lbl.dta.pkl',

    mode = 'train', # train or test, if test, make the is_load True
)
