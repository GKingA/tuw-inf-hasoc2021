from argparse import ArgumentParser
import pandas as pd
import numpy as np
import re
import emoji
from cleantext import clean
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def demojify(text, language='de'):
    return emoji.demojize(text, language=language)


def clean_other(text, language='de', mode='twitter'):
    if mode == 'twitter':
        text = re.sub('@[a-zA-Z0-9_]*', '[USER]', text)
        text = text.replace('#', '')
        text = text.replace('\n', ' ')
    else:
        text = text.replace('@USER', '[USER]')
        text = text.replace('@MODERATOR', '[MODERATOR]')
        text = text.replace('@MEDIUM', '[MEDIUM]')
    return clean(text, lower=False,
                 no_urls=True, no_numbers=False, no_currency_symbols=True,
                 replace_with_url='[URL]', replace_with_number='[NUMBER]', replace_with_currency_symbol='[CUR]',
                 lang=language)


def replace_others(text, text_processor):
    text = text_processor.spell_corrector.correct_text(text)
    text = text.strip().replace('\n', '<nl>')
    text = " ".join(text_processor.pre_process_doc(text))
    return text


def detect_dialect(sample, possible_delimiters=None):
    lines = sample.strip().split('\n')
    if possible_delimiters is None:
        possible_delimiters = [',', '\t', ';', ':', '|']
    elem_freq = list(map(lambda x, y: (y, x.count(y)), [lines[0] for _ in lines[0]], lines[0]))
    most_frequent = list(set([l[0] for l in filter(lambda x: x == max(elem_freq, key=lambda y: y[1]), elem_freq)]))
    possible_delimiters += most_frequent
    num_cols = 0
    delimiter = None
    has_header = False
    for d in possible_delimiters:
        first_line = len(re.sub("\".*\"|\'.*\'", "str", lines[0]).split(d))
        if len([1 for line in lines[1:] if len(re.sub("\".*\"|\'.*\'", "str", line).split(d)) != first_line]) == 0:
            num_cols = first_line
            delimiter = d
            break
    first_line = lines[0].split(delimiter)
    columns = np.array([re.sub("\".*\"|\'.*\'", "str", line).split(delimiter) for line in lines[1:]]).T
    for fl, col in zip(first_line, columns):
        if fl in col:
            continue
        mean = np.mean([len(c) for c in col])
        std = np.std([len(c) for c in col])
        if len(fl) < mean - std or len(fl) > mean + std:
            has_header = True
    return has_header, num_cols, delimiter


def read_csv(path, names, force_names=False):
    with open(path, 'r') as csv_file:
        string = csv_file.read()
        has_header, num_cols, delimiter = detect_dialect(string)
        if len(names) < num_cols and force_names:
            raise Exception(f"Not enough names given. The file has {num_cols} "
                            f"columns, the number of names given is {len(names)}")
        if has_header and force_names:
            first_line = string.split('\n')[0].strip()
            for idx, col in enumerate(first_line.split(delimiter)):
                if col == '' or col.startswith('Unnamed'):
                    names.insert(idx, f'Unnamed col: {idx}')
    if delimiter is None:
        quoting = 0
    else:
        quoting = 3 if delimiter not in ',.:;' else 0
    doublequote = quoting == 0
    if has_header and not force_names:
        return pd.read_csv(path, sep=delimiter, quoting=quoting, doublequote=doublequote)
    elif has_header:
        return pd.read_csv(path, sep=delimiter, header=0, names=names[:num_cols], quoting=quoting,
                           doublequote=doublequote)
    else:
        return pd.read_csv(path, sep=delimiter, header=None, names=names[:num_cols], quoting=quoting,
                           doublequote=doublequote)


def batch(df, batch_size):
    if batch_size == 0:
        return [df]
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]


def handle_data(path, batch_size, label_dict, binary_label_dict, data_type='twitter'):
    if data_type == 'hasoc':
        df = read_csv(path, names=['text_id', 'text', 'binary', 'labels', 'ID'], force_names=True)
    else:
        df = read_csv(path, names=['text', 'binary', 'labels', 'explicit'])
    if 'labels' in df:
        df.labels = df.labels.replace(label_dict)
    if 'binary' in df:
        df.binary = df.binary.replace(binary_label_dict)
    return batch(df, batch_size)


def handle_bin_data(path, batch_size, need_other, all_data=False, data_type='twitter'):
    if data_type == 'hasoc':
        distinct_datasets = {'OFFN': None, 'HATE': None, 'PRFN': None}
        distinct_data = {'OFFN': [], 'HATE': [], 'PRFN': []}
        df = read_csv(path, names=['text_id', 'text', 'binary', 'labels', 'ID'], force_names=True)
        other_name = 'NONE'
    else:
        distinct_datasets = {'ABUSE': None, 'INSULT': None, 'PROFANITY': None}
        distinct_data = {'ABUSE': [], 'INSULT': [], 'PROFANITY': []}
        df = read_csv(path, names=['text', 'binary', 'labels', 'explicit'])
        other_name = 'OTHER'
    for cat in distinct_datasets:
        if not all_data:
            distinct_datasets[cat] = df[df.labels == cat]
            other = int(len(distinct_datasets[cat]) / 3) if need_other else int(len(distinct_datasets[cat]) / 2)
            if need_other:
                distinct_datasets[cat] = distinct_datasets[cat].append(df[df.labels == other_name].sample(other))
            for other_cat in distinct_datasets:
                if other_cat == cat:
                    continue
                if len(df[df.labels == other_cat]) >= other:
                    distinct_datasets[cat] = distinct_datasets[cat].append(df[df.labels == other_cat].sample(other))
                else:
                    distinct_datasets[cat] = distinct_datasets[cat].append(df[df.labels == other_cat])
            distinct_datasets[cat].reset_index(inplace=True, drop=True)
            distinct_datasets[cat].binary = pd.Series([(cat == lab) * 1 for lab in distinct_datasets[cat].labels])
            distinct_datasets[cat] = distinct_datasets[cat].sample(frac=1)
        else:
            if need_other:
                distinct_datasets[cat] = df
            else:
                distinct_datasets[cat] = df[df.binary != other_name].reset_index(drop=True)
            if 'binary' in df:
                distinct_datasets[cat].binary = pd.Series([(cat == lab) * 1 for lab in distinct_datasets[cat].labels])
        distinct_data[cat] = batch(distinct_datasets[cat], batch_size)

    return distinct_data


def read_bin_data(path, batch_size, need_other=False, all_data=False, data_type='twitter'):
    if isinstance(path, str):
        train_path = path
        valid_path = None
    else:
        train_path = path[0]
        valid_path = path[1]
    data_lists = handle_bin_data(train_path, batch_size, need_other, all_data=all_data, data_type=data_type)
    if valid_path is not None:
        train = data_lists
        valid = handle_bin_data(valid_path, batch_size, need_other, all_data=True, data_type=data_type)
    else:
        if data_type == 'hasoc':
            train = {'OFFN': None, 'HATE': None, 'PRFN': None}
        else:
            train = {'ABUSE': None, 'INSULT': None, 'PROFANITY': None}
        valid = train.copy()
        for cat in train:
            train[cat], valid[cat] = train_test_split(data_lists[cat], test_size=0.25, random_state=42)
    return train, valid


def read_data(path, batch_size=8, drop_other=False, data_type='twitter'):
    if isinstance(path, str):
        train_path = path
        valid_path = None
    else:
        train_path = path[0]
        valid_path = path[1]
    if data_type == 'twitter':
        if drop_other:
            label_dict = {'ABUSE': 0, 'INSULT': 1, 'PROFANITY': 2}
        else:
            label_dict = {'OTHER': 0, 'ABUSE': 1, 'INSULT': 2, 'PROFANITY': 3}
        binary_label_dict = {'OTHER': 0, 'OFFENSE': 1}
    else:
        if drop_other:
            label_dict = {'OFFN': 0, 'HATE': 1, 'PRFN': 2}
        else:
            label_dict = {'NONE': 0, 'OFFN': 1, 'HATE': 2, 'PRFN': 3}
        binary_label_dict = {'NOT': 0, 'HOF': 1}
    data_list = handle_data(train_path, batch_size, label_dict, binary_label_dict, data_type=data_type)
    if valid_path is not None:
        train = data_list
        valid = handle_data(valid_path, batch_size, label_dict, binary_label_dict, data_type=data_type)
    else:
        train, valid = train_test_split(data_list, test_size=0.25, random_state=42)

    if drop_other:
        train_df = pd.concat(train)
        train_df = train_df[train_df.binary == 1]
        valid_df = pd.concat(valid)
        valid_df = valid_df[valid_df.binary == 1]
        train, valid = batch(train_df, batch_size), batch(valid_df, batch_size)

    return train, valid, label_dict, binary_label_dict


def read_toxic(path, batch_size=8, split=True):
    if isinstance(path, str):
        train_path = path
        valid_path = None
    else:
        train_path = path[0]
        valid_path = path[1]
    train_df = read_csv(train_path,
                        names=['comment_id', 'comment_text', 'Sub1_Toxic', 'Sub2_Engaging', 'Sub3_FactClaiming'],
                        force_names=True)
    if 'Sub1_Toxic' not in train_df:
        train = {'TOXIC': batch(pd.DataFrame(data={'text': train_df.comment_text,
                                                   'binary': pd.Series([0 for _ in range(len(train_df.comment_text))])}),
                                batch_size),
                 'ENGAGING': batch(pd.DataFrame(data={'text': train_df.comment_text,
                                                      'binary': pd.Series([0 for _ in range(len(train_df.comment_text))])}),
                                   batch_size),
                 'FACT': batch(pd.DataFrame(data={'text': train_df.comment_text,
                                                  'binary': pd.Series([0 for _ in range(len(train_df.comment_text))])}),
                               batch_size)}
    else:
        train = {'TOXIC': batch(pd.DataFrame(data={'text': train_df.comment_text, 'binary': train_df.Sub1_Toxic}),
                                batch_size),
                 'ENGAGING': batch(pd.DataFrame(data={'text': train_df.comment_text, 'binary': train_df.Sub2_Engaging}),
                                   batch_size),
                 'FACT': batch(pd.DataFrame(data={'text': train_df.comment_text, 'binary': train_df.Sub3_FactClaiming}),
                               batch_size)}
    if valid_path is None and split:
        valid = {}
        for cat in train:
            train[cat], valid[cat] = train_test_split(train[cat], test_size=0.25, random_state=42)
    elif valid_path is not None:
        valid_df = read_csv(valid_path,
                            names=['comment_id', 'comment_text', 'Sub1_Toxic', 'Sub2_Engaging', 'Sub3_FactClaiming'],
                            force_names=True)
        if 'Sub1_Toxic' not in valid_df:
            valid = {'TOXIC': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                       'binary': pd.Series([0 for _ in range(len(valid_df.comment_text))])}),
                                    batch_size),
                     'ENGAGING': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                          'binary': pd.Series([0 for _ in range(len(valid_df.comment_text))])}),
                                       batch_size),
                     'FACT': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                      'binary': pd.Series([0 for _ in range(len(valid_df.comment_text))])}),
                                   batch_size)}
        else:
            valid = {'TOXIC': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                       'binary': valid_df.Sub1_Toxic}),
                                    batch_size),
                     'ENGAGING': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                          'binary': valid_df.Sub2_Engaging}),
                                       batch_size),
                     'FACT': batch(pd.DataFrame(data={'text': valid_df.comment_text,
                                                      'binary': valid_df.Sub3_FactClaiming}),
                                   batch_size)}
    else:
        return train, None
    return train, valid


def shuffle(df_list):
    df = pd.concat(df_list)
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
    data_list = []
    for i in range(0, len(df) - 7, 8):
        data_list.append(df.iloc[i:i + 8])
    return data_list


def save_normalized(path, list_of_lines):
    with open(path, 'w') as normalized:
        normalized.write('\n'.join(list_of_lines))


def check_vocab(vocab_file, checker_file):
    words_outside = {}
    with open(vocab_file) as vocab:
        with open(checker_file) as checker:
            vocabulary = vocab.read().strip().split('\n')
            for line in checker:
                tokens = line.split()
                for token in tokens:
                    if token not in vocabulary:
                        if token not in words_outside:
                            words_outside[token] = 1
                        else:
                            words_outside[token] += 1
    return words_outside


def find_important(checker_file, pretrained):
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    with open(checker_file) as checker:
        for line in checker:
            tokens = tokenizer.tokenize(line)
            print(tokens)


def run_preprocessing(paths, output_paths, mode='twitter', language='de'):
    for path, out in zip(paths, output_paths):
        if path.endswith('xlsx'):
            df = pd.read_excel(path, engine='openpyxl')
        else:
            df = read_csv(path, names=['text', 'binary', 'labels', 'explicit'])
        texts = []
        text = 'text' if 'text' in df else 'comment_text'
        if text not in df:
            text = 'c_text'
        for line in df[text]:
            texts.append(clean_other(demojify(line, language=language), mode=mode).replace('\n', ' '))
        df[text] = texts
        df.to_csv(out, sep='\t', index=False)


def concat_for_train(paths, result, without):
    dfs = []
    for path in paths:
        df = read_csv(path, names=[])
        if 'comment_text' in df:
            df['text'] = df.comment_text
        if 'c_text' in df:
            df['text'] = df.c_text
        if 'Sub1_Toxic' in df:
            df['binary'] = df.Sub1_Toxic
        if 'task_1' in df:
            df['binary'] = df.task_1
        if 'task1' in df:
            df['binary'] = df.task1
        if 'task_2' in df:
            df['labels'] = df.task_2
        if 'task2' in df:
            df['labels'] = df.task2
        df.binary.replace('HOF', 'OFFENSE', inplace=True)
        df.binary.replace(1, 'OFFENSE', inplace=True)
        df.binary.replace('NOT', 'OTHER', inplace=True)
        df.binary.replace(0, 'OTHER', inplace=True)
        if 'labels' in df:
            df.labels.replace('NONE', 'OTHER', inplace=True)
            df.labels.replace('HATE', 'ABUSE', inplace=True)
            df.labels.replace('OFFN', 'INSULT', inplace=True)
            df.labels.replace('PRFN', 'PROFANITY', inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)
    if without is not None:
        df_negate = read_csv(without, ['text', 'binary', 'labels', 'explicit'])
        for text in df_negate.text:
            df = df[df.text != text]
    df.to_csv(result, sep='\t', index=False)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--to_normalize', nargs='+')
    argparser.add_argument('--normalized_path', nargs='+')
    argparser.add_argument('--mode', choices=['facebook', 'twitter'], default='twitter')
    argparser.add_argument('--language', choices=['en', 'de'], default='de')
    argparser.add_argument('--concat')
    argparser.add_argument('--without')
    args = argparser.parse_args()
    if args.concat is None:
        run_preprocessing(args.to_normalize, args.normalized_path, args.mode, args.language)
    else:
        concat_for_train(args.normalized_path, args.concat, args.without)
