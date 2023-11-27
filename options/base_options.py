import argparse
import os

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # data augmentation
        parser.add_argument('--dataroot', type=str, help='path to relative dataroot')
        parser.add_argument('--output_dir', type=str, help='output path')
        parser.add_argument('--protocol_dir', type=str, default='', help='path to where the protocols are stored')
        parser.add_argument('--protocol_name', type=str, default='', help='name of protocol, e.g. default_stylegan2')
        parser.add_argument('--architecture', type=str, default='one_stream', help='name of architecture: options, one_stream or two_stream')
        parser.add_argument('--is_aligned', action='store_true', help='should be true if the images are already resolution 224x224 and the face images has been aligned')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0. use -1 for CPU')
        parser.add_argument('--spectra', type=str, default="rgb", help='specify the spectra to use. "rgb, fft or multi"')
        return parser


    def _check_args(self, args):
        raise NotImplementedError("Please Implement this method")

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        args =  parser.parse_args()
        self._check_args(args)
        return args

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        file_name = os.path.join(opt.output_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        if print_options:
            self.print_options(opt)

        self.opt = opt
        return self.opt