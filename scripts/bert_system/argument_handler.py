from argparse import ArgumentParser
import json
import os


def load_config(args):
    if args.config is not None:
        config = json.load(open(args.config))
        for setting in args.__dict__:
            if setting in config:
                args.__dict__[setting] = config[setting]
    return args


def parse_arguments():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data', '-d', help='Path to the dataset.')
    arg_parser.add_argument('--epoch', '-ep', help='The number of training epochs.', default=10, type=int)
    arg_parser.add_argument('--embedding', '-e', help='Which embedding to use.', choices=['MUSE', 'BERT'],
                            default='BERT')
    arg_parser.add_argument('--muse_path', help='The path to the appropriate MUSE embedding.',
                            default=os.path.join(os.path.dirname(__file__), '../../MUSE'))
    arg_parser.add_argument('--dont_use_lstm', help='If set, the model will only use dense layers and no lstm.',
                            action='store_true', default=False)
    arg_parser.add_argument('--language', '-l', help='The language of the training data. '
                                                     'Used only with MUSE embedding.',
                            default='en')
    arg_parser.add_argument('--draw_heatmap', '-dh', help='If set, the program will draw a confusion matrix heatmap '
                                                          'after every epoch. Otherwise it will only '
                                                          'print out the matrix.',
                            action='store_true', default=False)
    arg_parser.add_argument('--balance', '-b', help='If set, the program will try to eliminate some of the elements '
                                                    'of the highest instance count category.',
                            action='store_true', default=False)
    arg_parser.add_argument('--gpu', help='GPU id to use. If none, set it to -1, '
                                          'or leave as is, as the default is to use cpu.', default=-1, type=int)
    arg_parser.add_argument('--evaluate', help='If set, the model will not be trained, only evaluated.',
                            action='store_true', default=False)
    arg_parser.add_argument('--multi_label', help='If set, the model can predict multiple labels for an input.',
                            action='store_true', default=False)
    arg_parser.add_argument('--class_weights', help='What function to use to weigh the classes relative to each other. '
                                                    'If none given, the model won\'t use class weights.',
                            choices=['log', 'avg', 'scikit'], default=None)
    arg_parser.add_argument('--model', '-m', help='The model architecture to use.',
                            choices=['BaseBertClassifier', 'SentimentClassifier', 'LogisticModel', 'SentimentKeras'],
                            default='SentimentClassifier')
    arg_parser.add_argument('--save_model', help='The path to save the model to.', default="model")
    arg_parser.add_argument('--load_model', help='The path to load the model from. '
                                                 'If not set, the model is initialized normally.', default=None)
    arg_parser.add_argument('--random', '-r', help='Whether to randomly cut the train to train and dev.',
                            action='store_true', default=False)
    arg_parser.add_argument('--ratio', '-rat', help='The ratio of the test in the whole data.', type=float, default=0.2)
    arg_parser.add_argument('--chunk_size', '-ch', help='The chunk size to read in at once.', type=int, default=None)
    arg_parser.add_argument('--categories', '-cat', help='The categories to predict.', nargs='+', default=None)
    arg_parser.add_argument("--config", "-c", help='The config file path.', default=None)

    args = arg_parser.parse_args()

    return load_config(args)
