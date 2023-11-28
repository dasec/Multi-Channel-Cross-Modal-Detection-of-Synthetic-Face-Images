from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path', type=str, help='path to pretrained_model', required=True)
        parser.add_argument('--eval_split', type=str, help='name of csv containing eval data', default='eval.csv')
        parser.add_argument('--result_file', type=str, help='name of file to store resutls in, will be stored relative to the output dir', default='results.csv')

        return parser
    
    def _check_args(self, args):
        pass