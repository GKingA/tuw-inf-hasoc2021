import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from argparse import ArgumentParser
from read_data import read_csv


def create_toxic_result(path, expected, out, toxic_category, test=False):
    if not test:
        df_true = read_csv(expected, names=['comment_id', 'toxic', 'engaging', 'fact'],
                           force_names=True)
    else:
        df_true = read_csv(expected, names=['comment_id', 'comment_text'], force_names=True)
    df_binary = read_csv(path, names=['text', 'OTHER', 'CATEGORY'], force_names=True)
    df_binary["result"] = pd.Series(
        [1 if category >= other else 0 for (category, other) in zip(df_binary.CATEGORY, df_binary.OTHER)])
    result = pd.concat([df_binary, df_true], axis=1)
    if not test:
        print(confusion_matrix(result[toxic_category].tolist(), result.result.tolist()))
        print(classification_report(result[toxic_category].tolist(), result.result.tolist()))
    comment_text = 'text' if 'text' in result else 'comment_text'
    result.to_csv(out, index=False, sep='\t', columns=['comment_id', comment_text, 'result'])
    return classification_report(result[toxic_category].tolist(), result.result.tolist(), output_dict=True)


def create_binary_result(path, expected, out, hasoc, test=True):
    if hasoc:
        df_true = read_csv(expected, names=['id', 'c_text', 'binary', 'labels', 'additional'], force_names=True)
    else:
        df_true = read_csv(expected, names=['text', 'binary', 'labels'])
    df_binary = read_csv(path, names=['text', 'OTHER', 'OFFENSE'])
    off = 'OFFENSE' if 'OFFENSE' in df_binary else 'HOF'
    oth = 'OTHER' if 'OTHER' in df_binary else 'NOT'
    try:
        df_binary["result"] = pd.Series(
            [off if offense >= other else oth for (offense, other) in zip(df_binary[off], df_binary[oth])])
    except KeyError:
        return {'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0}}
    result = pd.concat([df_binary, df_true], axis=1)
    if not test:
        print(confusion_matrix(result.binary.tolist(), result.result.tolist()))
        print(classification_report(result.binary.tolist(), result.result.tolist()))
        cr = classification_report(result.binary.tolist(), result.result.tolist(), output_dict=True)
    else:
        cr = None
    result["binary"] = result["result"]
    if 'labels' in result:
        result.to_csv(out, index=False, sep='\t', columns=['text', 'binary', 'labels'])
    else:
        result.to_csv(out, index=False, sep=',', columns=['id', 'binary', 'c_text'])
    return cr


def determine_offense(df, dont_use_binary):
    off = 'OFFENSE' if 'OFFENSE' in df else 'HOF'
    oth = 'OTHER' if 'OTHER' in df else 'NOT'
    if dont_use_binary:
        return pd.Series(['OTHER' if abuse < 0.5 and profanity < 0.5 and insult < 0.5 else
                          ('PROFANITY' if profanity > abuse and profanity > insult else
                           ('INSULT' if insult > abuse else 'ABUSE'))
                          for (profanity, abuse, insult) in
                          zip(df.PROFANITY, df.ABUSE, df.INSULT)])
    return pd.Series(['OTHER' if other > offense else
                      ('PROFANITY' if profanity > abuse and profanity > insult else
                       ('INSULT' if insult > abuse else 'ABUSE'))
                      for (other, offense, profanity, abuse, insult) in
                      zip(df[oth], df[off], df.PROFANITY, df.ABUSE, df.INSULT)])


def find_best_binary_result(binary, abuse, insult, profanity, expected, epochs=10):
    df_true = read_csv(expected, names=['text', 'binary', 'labels'])
    best = 0
    best_tuple = ()
    results = {}
    for b in range(epochs):
        for a in range(epochs):
            for i in range(epochs):
                for p in range(epochs):
                    df_binary = read_csv(binary.format(b), names=['text', 'OTHER', 'OFFENSE'])
                    ab = read_csv(abuse.format(a), names=['text', 'OTHER', 'ABUSE'])
                    ab.drop('OTHER', axis=1, inplace=True)
                    ins = read_csv(insult.format(i), names=['text', 'OTHER', 'INSULT'])
                    ins.drop('OTHER', axis=1, inplace=True)
                    pro = read_csv(profanity.format(p), names=['text', 'OTHER', 'PROFANITY'])
                    pro.drop('OTHER', axis=1, inplace=True)
                    ab_in = ab.merge(ins, how='inner')
                    df = ab_in.merge(pro, how='inner')
                    df = df.merge(df_binary, how="right")
                    df.fillna(value=0, inplace=True)
                    df["result"] = determine_offense(df)
                    result = pd.merge(df, df_true, how='inner')
                    print(confusion_matrix(result.labels.tolist(), result.result.tolist()))
                    print(classification_report(result.labels.tolist(), result.result.tolist()))
                    results[(b, a, i, p)] = \
                        classification_report(result.labels.tolist(), result.result.tolist(), output_dict=True)
                    if results[(b, a, i, p)]['macro avg']['f1-score'] > best:
                        best = results[(b, a, i, p)]['macro avg']['f1-score']
                        best_tuple = (b, a, i, p)
    print(results[best_tuple])
    return best_tuple


def create_all_binary_result(binary, abuse, insult, profanity, expected, out, test, dont_use_binary):
    if 'HASOC' in expected:
        df_true = read_csv(expected, names=['id', 'text', 'binary', 'labels', 'explicit'], force_names=True)
        if not test:
            df_true.labels.replace('HATE', 'ABUSE', inplace=True)
            df_true.labels.replace('OFFN', 'INSULT', inplace=True)
            df_true.labels.replace('PRFN', 'PROFANITY', inplace=True)
            df_true.labels.replace('NONE', 'OTHER', inplace=True)
    else:
        df_true = read_csv(expected, names=['text', 'binary', 'labels'])
    df_binary = read_csv(binary, names=['text', 'OTHER', 'OFFENSE'], force_names=True)
    ab = read_csv(abuse, names=['text_1', 'OTHER', 'ABUSE'], force_names=True)
    ab.drop('OTHER', axis=1, inplace=True)
    ins = read_csv(insult, names=['text_2', 'OTHER', 'INSULT'], force_names=True)
    ins.drop('OTHER', axis=1, inplace=True)
    pro = read_csv(profanity, names=['text_3', 'OTHER', 'PROFANITY'], force_names=True)
    pro.drop('OTHER', axis=1, inplace=True)

    df = pd.concat([ab, ins, pro, df_binary], axis=1)
    df.fillna(value=0, inplace=True)
    df["result"] = determine_offense(df, dont_use_binary)
    if not test:
        print(confusion_matrix(df_true.labels.tolist(), df.result.tolist()))
        print(classification_report(df_true.labels.tolist(), df.result.tolist()))
    if 'HASOC' in expected:
        df.result.replace('ABUSE', 'HATE', inplace=True)
        df.result.replace('INSULT', 'OFFN', inplace=True)
        df.result.replace('PROFANITY', 'PRFN', inplace=True)
        df.result.replace('OTHER', 'NONE', inplace=True)
    result = pd.concat([df, df_true], axis=1)
    result.to_csv(out, sep=',', index=False, columns=['id', 'result'])


def create_categorical_result(categorical, binary, expected, out):
    if 'HASOC' in expected:
        df_true = read_csv(expected, names=['id', 'text', 'binary', 'labels', 'explicit'], force_names=True)
        df_true.labels.replace('HATE', 'ABUSE', inplace=True)
        df_true.labels.replace('OFFN', 'INSULT', inplace=True)
        df_true.labels.replace('PRFN', 'PROFANITY', inplace=True)
        df_true.labels.replace('NONE', 'OTHER', inplace=True)
    else:
        df_true = read_csv(expected, names=['text', 'binary', 'labels'])
    df_binary = read_csv(binary, names=['text', 'OTHER', 'OFFENSE'], force_names=True)
    df_categorical = read_csv(categorical, names=['text', 'OTHER', 'INSULT', 'ABUSE', 'PROFANITY'], force_names=True)
    df_categorical["result"] = df_categorical.drop('text', axis=1).idxmax(axis=1)
    result = pd.concat([df_categorical, df_true], axis=1)
    print(confusion_matrix(result.labels.tolist(), result.result.tolist()))
    print(classification_report(result.labels.tolist(), result.result.tolist()))
    result.to_csv(out, sep='\t', index=False, columns=['text', 'result'])


def voting(paths, out, weights=None):
    df = None
    for i, path in enumerate(paths):
        f = open(path)
        first_line = f.readline()
        f.close()
        if 'comment_id' in first_line:
            df_right = read_csv(path, names=['comment_id', f'text_{i}', f'label_{i}'], force_names=True)
        else:
            df_right = read_csv(path, names=[f'text_{i}', f'label_{i}'], force_names=True)
        df = df_right if df is None else pd.concat([df, df_right], axis=1)
    if weights is None:
        df["vote"] = (pd.concat([df[f'label_{i}'] for i in range(len(paths))], axis=1).mean(axis=1) >= 0.5) * 1
        df["vote_2"] = (pd.concat([df[f'label_{i}'] for i in range(len(paths))], axis=1).sum(axis=1) >= 1) * 1
    df.to_csv(out, sep='\t', index=False, columns=['comment_id', 'text_0', 'vote_2'])


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--toxic_category', choices=["toxic", "engaging", "fact", "none"], default="none")
    argparser.add_argument('--binary')
    argparser.add_argument('--abuse')
    argparser.add_argument('--insult')
    argparser.add_argument('--profanity')
    argparser.add_argument('--categorical')
    argparser.add_argument('--expected')
    argparser.add_argument('--vote', nargs='+')
    argparser.add_argument('--weights', nargs='+', type=float)
    argparser.add_argument('--out', required=True)
    argparser.add_argument('--find', action='store_true')
    argparser.add_argument('--test', action='store_true')
    argparser.add_argument('--hasoc', action='store_true')
    argparser.add_argument('--dont_use_binary', action='store_true')
    args = argparser.parse_args()
    if args.vote is not None:
        voting(args.vote, args.out, args.weights)
    elif args.categorical is not None:
        create_categorical_result(args.categorical, args.binary, args.expected, args.out)
    elif args.toxic_category != "none":
        create_toxic_result(args.binary, args.expected, args.out, args.toxic_category, test=args.test)
    elif args.binary is not None and args.abuse is not None and args.insult is not None and args.profanity is not None:
        if args.find:
            indices = find_best_binary_result(args.binary, args.abuse, args.insult, args.profanity, args.expected)
            create_all_binary_result(args.binary.format(indices[0]), args.abuse.format(indices[1]),
                                     args.insult.format(indices[2]), args.profanity.format(indices[3]),
                                     args.expected, args.out, args.test, args.dont_use_binary)
        else:
            create_all_binary_result(args.binary, args.abuse, args.insult, args.profanity, args.expected, args.out,
                                     args.test, args.dont_use_binary)
    else:
        create_binary_result(args.binary, args.expected, args.out, hasoc=args.hasoc, test=args.test)
