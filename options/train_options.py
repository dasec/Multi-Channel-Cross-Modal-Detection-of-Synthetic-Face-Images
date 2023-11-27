from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_split', type=str, default='train.csv', help='csv file containing the training information')
        parser.add_argument('--val_split', type=str, default='dev.csv', help='csv file containing the validation information')
        parser.add_argument('--loss', type=str, default='bce', help='loss to use: bce or cmfl')
        parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.00001, help='learning rate')
        parser.add_argument('--seed', type=int, default=4, help='seed to use for torch')
        parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
        parser.add_argument('--save_interval', type=int, default=5, help='interval for saving model')
        parser.add_argument('--augment_chance', type=float, default=0.5, help='specifies the chance of applying gaussian blurring and jpeg compression. If 0, no augmentation will take place')
        parser.add_argument('--augment_val', action='store_true', help='specifies if the validation set should be augmented also, if false only train will undergo validation')
        parser.add_argument('--pretrained', action='store_true', help='specifies if pretrained weights should be used')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size to use for training or testing')
        return parser


    def _check_args(self, args):
        assert args.augment_chance < 1 and args.augment_chance >= 0 # augment chance has to be between [0 and 1