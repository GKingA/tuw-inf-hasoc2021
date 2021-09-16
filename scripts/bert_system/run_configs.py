import os
import subprocess
import json
from voter import create_toxic_result, create_binary_result
from argparse import ArgumentParser


def list_configs(path):
    files = []
    if not os.path.isdir(path):
        raise Exception(f"The path given is not valid. Path: {path}")
    for config_file in os.listdir(path):
        if config_file.endswith('.json') and 'train' in config_file:
            files.append(os.path.join(path, config_file))
    print(files)
    return files


def make_multilingual(files):
    all_files = files.copy()
    for file_ in files:
        path = '.'.join(file_.split('.')[:-1])
        extension = file_.split('.')[-1]
        config = json.load(open(file_))
        new_config = config.copy()
        if new_config['bert_model'] != 'bert-base-multilingual-cased':
            new_config['bert_model'] = 'bert-base-multilingual-cased'
            if '_whole_data' in path:
                new_path = path[:-len('_whole_data')] + '_multilingual_whole_data'
                new_config['model_path'] = new_config['model_path'][:-len('_whole_data')] + '_multilingual_whole_data'
            else:
                new_path = path + '_multilingual'
                new_config['model_path'] = new_config['model_path'] + '_multilingual'
            new_path = '.'.join([new_path, extension])
            with open(new_path, 'w') as multi_path:
                json.dump(new_config, multi_path, indent=4)
            all_files.append(new_path)
    return all_files


def make_new_configs(best_epoch, config, test_file_path, test, train_config, cat_name, ext, test_dir):
    for epoch, num in best_epoch.items():
        new_config = config.copy()
        new_config['data_path'] = [test_file_path, test_file_path]
        new_config['model_path'] = new_config['model_path'] + f'_whole_data_epoch{num[1]}'
        new_config['train_test'] = 'predict'
        new_config['predict_path'] = f"predicted/{test}_{train_config}_{cat_name}_{epoch}.csv"
        # new_config['mode'] = 'binary'
        # if '{}' in new_config['model_path']:
        new_config['model_path'] = new_config['model_path'].format(cat_name)  # 'TOXIC')
        if test == 'GermEval21':
            new_config['predict_path'] = f"predicted/NLE/{test}_{train_config}_{{}}_{epoch}.csv"
            new_config['mode'] = 'binary_categories'
            new_config['data_type'] = 'facebook'
        elif 'HASOC' in test:
            new_config['data_type'] = 'hasoc'
        else:
            new_config['data_type'] = 'twitter'
        filename = train_config.replace('train', 'test') + f"_{epoch}.{ext}"
        if cat_name in new_config['model_path']:
            filename = train_config.replace('train', 'test') + f"_{cat_name}_{epoch}.{ext}"
        with open(os.path.join(test_dir, filename), 'w') as test_config:
            json.dump(new_config, test_config, indent=4)


def run_configs(files, test_path):
    with open(test_path) as tp:
        data_files = json.load(tp)
    base_dir = os.path.dirname(files[0])

    for file_ in files:
        print(file_)
        output = subprocess.run(['python3', 'train_model.py', '--config', file_],
                                stdout=subprocess.PIPE)
        print(output)
        train_config = '.'.join(os.path.basename(file_).split('.')[:-1])
        ext = file_.split('.')[-1]
        config = json.load(open(file_))
        if 'whole_data' in file_:
            continue
        best_epoch = {'macro_F1': [0, 0], 'toxic_F1': [0, 0], 'toxic_precision': [0, 0]}
        best_epochs = {'ABUSE': {'macro_F1': [0, 0], 'toxic_F1': [0, 0], 'toxic_precision': [0, 0]},
                       'PROFANITY': {'macro_F1': [0, 0], 'toxic_F1': [0, 0], 'toxic_precision': [0, 0]},
                       'INSULT': {'macro_F1': [0, 0], 'toxic_F1': [0, 0], 'toxic_precision': [0, 0]}}
        epochs = output.stdout.decode("utf-8").replace('\'', '\"').split("\n")
        for i, epoch in enumerate(epochs):
            if not epoch.startswith('{'):
                continue
            epoch_data = json.loads(epoch)
            cat_name = 'OFFENSE' if 'OFFENSE' in epoch_data else \
                ('TOXIC' if 'TOXIC' in epoch_data else ('HOF' if 'HOF' in epoch_data else None))
            if cat_name is None:
                cat_name = 'ABUSE' if 'ABUSE' in epoch_data else \
                    ('PROFANITY' if 'PROFANITY' in epoch_data else ('INSULT' if 'INSULT' in epoch_data else None))
                if cat_name is None:
                    continue
            if cat_name in best_epochs:
                if epoch_data['macro avg']['f1-score'] > best_epochs[cat_name]['macro_F1'][0]:
                    best_epochs[cat_name]['macro_F1'][0] = epoch_data['macro avg']['f1-score']
                    best_epochs[cat_name]['macro_F1'][1] = i % 10
                if epoch_data[cat_name]['f1-score'] > best_epochs[cat_name]['toxic_F1'][0]:
                    best_epochs[cat_name]['toxic_F1'][0] = epoch_data[cat_name]['f1-score']
                    best_epochs[cat_name]['toxic_F1'][1] = i % 10
                if epoch_data[cat_name]['precision'] > best_epochs[cat_name]['toxic_precision'][0]:
                    best_epochs[cat_name]['toxic_precision'][0] = epoch_data[cat_name]['precision']
                    best_epochs[cat_name]['toxic_precision'][1] = i % 10
            else:
                if epoch_data['macro avg']['f1-score'] > best_epoch['macro_F1'][0]:
                    best_epoch['macro_F1'][0] = epoch_data['macro avg']['f1-score']
                    best_epoch['macro_F1'][1] = i
                if epoch_data[cat_name]['f1-score'] > best_epoch['toxic_F1'][0]:
                    best_epoch['toxic_F1'][0] = epoch_data[cat_name]['f1-score']
                    best_epoch['toxic_F1'][1] = i
                if epoch_data[cat_name]['precision'] > best_epoch['toxic_precision'][0]:
                    best_epoch['toxic_precision'][0] = epoch_data[cat_name]['precision']
                    best_epoch['toxic_precision'][1] = i
        for test, test_file_path in data_files.items():
            test_dir = os.path.join(base_dir, test)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            if cat_name not in best_epochs:
                make_new_configs(best_epoch, config, test_file_path, test, train_config, cat_name, ext, test_dir)
            else:
                for (cat_name, best_epoch) in best_epochs.items():
                    make_new_configs(best_epoch, config, test_file_path, test, train_config, cat_name, ext, test_dir)
    for test in data_files:
        test_dir = os.path.join(base_dir, test)
        for conf in os.listdir(test_dir):
            conf_path = os.path.join(test_dir, conf)
            print(conf_path)
            subprocess.run(['python3', 'train_model.py', '--config', conf_path])


def vote(predicted_path, test_path):
    with open(test_path) as tp:
        data_files = json.load(tp)
    results = []
    for pred in os.listdir(predicted_path):
        pred_path = os.path.join(predicted_path, pred)
        print(pred)
        if pred.split('_')[0] == 'GermEval21':
            output = create_toxic_result(pred_path, data_files['GermEval21'],
                                         os.path.join(predicted_path, 'result', pred), 'toxic')
        elif pred.startswith('HASOC'):
            output = create_binary_result(pred_path, data_files['_'.join([pred.split('_')[0], pred.split('_')[1]])],
                                          os.path.join(predicted_path, 'result', pred), hasoc=True)
        elif pred.split('_')[0] in data_files:
            output = create_binary_result(pred_path, data_files[pred.split('_')[0]],
                                          os.path.join(predicted_path, 'result', pred), hasoc=False)
        else:
            continue
        off_cat = '1' if '1' in output else ('HOF' if 'HOF' in output else ('OFFENSE' if 'OFFENSE' in output else None))
        if off_cat is None:
            off_cat = "OFFENSE"
            output["OFFENSE"] = {'precision': 0, 'recall': 0, 'f1-score': 0}
        no_cat = '0' if '0' in output else ('NOT' if 'NOT' in output else ('OTHER' if 'OTHER' in output else None))
        if no_cat is None:
            no_cat = "OTHER"
            output["OTHER"] = {'precision': 0, 'recall': 0, 'f1-score': 0}
        results.append(f'&{pred}\n'
                       f'&{round(output[off_cat]["precision"]*100, 1)}'
                       f'&{round(output[off_cat]["recall"]*100, 1)}'
                       f'&{round(output[off_cat]["f1-score"]*100, 1)}\n'
                       f'&{round(output[no_cat]["precision"]*100, 1)}'
                       f'&{round(output[no_cat]["recall"]*100, 1)}'
                       f'&{round(output[no_cat]["f1-score"]*100, 1)}\n'
                       f'&{round(output["macro avg"]["precision"]*100, 1)}'
                       f'&{round(output["macro avg"]["recall"]*100, 1)}'
                       f'&{round(output["macro avg"]["f1-score"]*100, 1)}\n\n')

    with open("latex_results.txt", 'w') as latex:
        latex.write('\n'.join(sorted(results)))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'result'], default='train',
                            help='Train models (train) and run the best of them on the test files, '
                                 'or generate predictions (result) using the output files of the train mode.')
    arg_parser.add_argument('--configs', default='configs', help='The path to the training config files.')
    arg_parser.add_argument('--multilingual', action='store_true', help='Generate multilingual train configs.')
    arg_parser.add_argument('--test_path', help='Json file containing the test files.', default='test.json')
    args = arg_parser.parse_args()
    if args.mode == 'train':
        train_configs = list_configs(args.configs)
        if args.multilingual:
            train_configs = make_multilingual(train_configs)
        run_configs(train_configs, args.test_path)
    else:
        vote(args.configs, args.test_path)
