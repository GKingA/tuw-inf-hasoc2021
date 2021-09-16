import torch
from torch import nn
import copy
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, BertForPreTraining, BertModel
from read_data import read_data, read_bin_data, read_toxic, shuffle
import pandas as pd
from argparse import ArgumentParser
from argument_handler import load_config
from data_preprocessing import compute_class_weights
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 1234

torch.manual_seed(SEED)


class PretrainedClassifier(nn.Module):
    # https://github.com/huggingface/transformers/issues/2483.
    def delete_layers(self, num_layers_to_delete):
        old_module_list = self.bert_model.encoder.layer
        new_module_list = nn.ModuleList()
        for i in range(0, len(old_module_list) - num_layers_to_delete):
            new_module_list.append(old_module_list[i])
        copy_of_model = copy.deepcopy(self.bert_model)
        copy_of_model.encoder.layer = new_module_list
        return copy_of_model


class PretrainedBinary(PretrainedClassifier):
    def __init__(self, num_class=2, bert_model='bert-base-german-cased', num_layers_to_delete=3, device='cpu', weights=None, freeze=False):
        super(PretrainedBinary, self).__init__()
        self.num_class = num_class
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model).to(device)
        if freeze:
            for p in self.bert_model.base_model.parameters():
                p.requires_grad = False
        self.bert_model = self.delete_layers(num_layers_to_delete=num_layers_to_delete)
        self.lstm = nn.LSTM(768, 256, batch_first=True).to(device)
        self.out = nn.Linear(256, num_class).to(device)
        self.softmax = nn.Softmax(dim=1)
        if weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, text, labels=None, device='cpu'):
        tokenized = self.tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        embedding = self.bert_model(input_ids, attention_mask=attention_mask)
        out, hidden = self.lstm(embedding[1].unsqueeze(1))
        linear_out = self.out(out.squeeze()).view(len(labels), self.num_class)
        return self.loss_fct(linear_out, labels.to(device)), linear_out


class PretrainedOffense(PretrainedClassifier):
    def __init__(self, num_class=3, bert_model='bert-base-german-cased', num_layers_to_delete=2, device='cpu', weights=None, freeze=False):
        super(PretrainedOffense, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model).to(device)
        if freeze:
            for p in self.bert_model.base_model.parameters():
                p.requires_grad = False
        self.bert_model = self.delete_layers(num_layers_to_delete=num_layers_to_delete)
        self.out = nn.Linear(768, num_class).to(device)
        self.softmax = nn.Softmax(dim=1)
        if weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss().to(device)

    def forward(self, text, labels=None, device='cpu'):
        tokenized = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        embedding = self.bert_model(input_ids, attention_mask=attention_mask)
        linear_out = self.out(embedding[1])
        #linear_out = self.softmax(linear_out)
        return self.loss_fct(linear_out, labels.to(device)), linear_out


class BertMasked(nn.Module):
    def __init__(self, bert_model='bert-base-german-cased', device='cpu'):
        super(BertMasked, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertForPreTraining.from_pretrained(bert_model).to(device)

    def forward(self, text, device='cpu'):
        tokenized = self.tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        labels = input_ids
        return self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)


class BaseBertModel(nn.Module):
    def __init__(self, num_class, bert_model='bert-base-german-cased', device='cpu', weights=None, freeze=False):
        super(BaseBertModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_class).to(device)
        if freeze:
            for p in self.bert_model.base_model.parameters():
                p.requires_grad = False
        if weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, text, labels=None, device='cpu'):
        tokenized = self.tokenizer(text, truncation=True, max_length=128, padding=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        logits = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels.to(device))[1]
        loss = self.loss_fct(logits.view(-1, self.bert_model.num_labels), labels.view(-1).to(device))
        return loss, logits


class MachineLearning(nn.Module):
    def __init__(self, num_class, device='cpu', weights=None, corpus=None):
        super(MachineLearning, self).__init__()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.matrix = None
        if corpus is not None:
            self.init_vectorizer(corpus)
        self.lstm = nn.LSTM(len(self.vectorizer.vocabulary_), 750).to(device)
        self.out = nn.Linear(750, num_class).to(device)
        if weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device))
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def init_vectorizer(self, corpus):
        self.matrix = self.vectorizer.fit(corpus)

    def forward(self, text, labels=None, device='cpu'):
        x = torch.tensor(self.matrix.transform(text).toarray()).view(len(text), 1, -1).float().to(device)
        x, _ = self.lstm(x)
        out = self.out(x.view(len(text), -1))
        loss = self.loss_fct(out, labels.view(-1).to(device))
        return loss, out


def train(model, data, optimizer, label_dict, device='cpu'):
    train_loss = 0
    train_acc = 0
    predicted = []
    labels = []
    model.train()
    for i, batch in enumerate(data):
        optimizer.zero_grad()
        if len(label_dict) == 2:
            label = torch.LongTensor(batch.binary.values.tolist())
            labels += batch.binary.values.tolist()
        else:
            label = torch.LongTensor(batch.labels.values.tolist())
            labels += batch.labels.values.tolist()
        text = batch.text.values.tolist()
        output = model(text, labels=label, device=device)
        loss = output[0]
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output[1].argmax(axis=1)
        train_acc += torch.eq(pred, label.to(device)).sum()
        predicted += pred.tolist()


def evaluate(model, data, label_dict, device='cpu'):
    valid_loss = 0
    valid_acc = 0
    predicted = []
    labels = []
    model.eval()
    batch_len = 0
    num_data = 0
    for i, batch in enumerate(data):
        num_data += len(batch)
        if len(label_dict) == 2:
            label = torch.LongTensor(batch.binary.values.tolist())
            labels += batch.binary.values.tolist()
        else:
            label = torch.LongTensor(batch.labels.values.tolist())
            labels += batch.labels.values.tolist()
        text = batch.text.values.tolist()
        output = model(text, labels=label, device=device)
        loss = output[0]
        valid_loss += loss.item()
        pred = output[1].argmax(axis=1)
        valid_acc += torch.eq(pred, label.to(device)).sum()
        predicted += pred.tolist()
    valid_loss = torch.true_divide(valid_loss, len(data))
    valid_acc = torch.true_divide(valid_acc, num_data)
    cr = classification_report(labels, predicted, target_names=label_dict.values(), output_dict=True)
    print(cr)
    return (None, cr), valid_acc, valid_loss


def predict(model, data, label_dict, file_path=None, device='cpu'):
    predicted = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    if file_path is None:
        file_path = "predictions.tsv"
    out_file = open(file_path, 'w')
    labels = '\t'.join([lab for lab in label_dict.values()])
    out_file.write(f'text\t{labels}\n')
    for i, batch in enumerate(data):
        if len(label_dict) == 2 and 'binary' in batch:
            label = torch.LongTensor(batch.binary.values.tolist())
        elif 'labels' in batch:
            label = torch.LongTensor(batch.labels.values.tolist())
        else:
            label = torch.LongTensor([0 for _ in batch.text])
        text = batch.text.values.tolist()
        output = model(text, labels=label, device=device)
        predicted += output[1].argmax(axis=1).tolist()
        probs = softmax(output[1])
        for t, prob in zip(text, probs):
            prob_str = '\t'.join([str(p) for p in prob.detach().cpu().numpy()])
            out_file.write(f'{t}\t{prob_str}\n')
    out_file.close()
    return predicted


def predict_two_models(model_binary, model_classes, data, label_dict, binary_label_dict, device='cpu'):
    binary = predict(model_binary, data, binary_label_dict, device=device)
    og_df = pd.concat(data)
    df = pd.concat(data)
    df.binary = binary
    df.labels = [0 for _ in range(len(binary))]
    df_classes = df[df.binary == 1]
    classes = [df_classes.iloc[i:i + 8] for i in range(0, len(df_classes), 8)]
    classes_predicted = predict(model_classes, classes, label_dict, device=device)
    df_classes.labels = classes_predicted
    df.update(df_classes)


def training_iteration(epochs, model, train_data, valid_data, optimizer, labels, early_stopping, model_path, device='cpu'):
    prev_macro, prev_micro, prev_loss, prev_acc = 0, 0, float('inf'), 0
    for epoch in range(epochs):
        # print(f"\nEpochs: {epoch}/{epochs}")
        train(model, train_data, optimizer, labels, device=device)
        (_, stats), acc, loss = evaluate(model, valid_data, labels, device=device)
        if early_stopping != 'none':
            if (loss > prev_loss and early_stopping == 'loss') or (acc < prev_acc and early_stopping == 'acc') \
                    or (stats['MACRO AVG']['f1'] > prev_macro and early_stopping == 'macro_f1')\
                    or (stats['MICRO AVG']['f1'] > prev_macro and early_stopping == 'macro_f1'):
                break
            prev_macro, prev_micro, prev_loss, prev_acc = stats['MACRO AVG']['F1'], stats['MICRO AVG']['F1'], loss, acc
        train_data = shuffle(train_data)
        if model_path is not None:
            torch.save(model, f'{model_path}_epoch{epoch}')


def init_optim(model, optim, weight_decay, lr):
    if lr is not None:
        optimizer = optimizer_dict[optim](model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optimizer_dict[optim](model.parameters(), weight_decay=weight_decay)

    return optimizer


def run_in_mode(train_test, model, train_data, valid_data, optim, lr, epochs, model_path, labels, early_stopping,
                predict_path, device):
    if train_test == 'train':
        optimizer = init_optim(model, optim, 1e-5, lr)
        training_iteration(epochs, model, train_data, valid_data, optimizer, labels, early_stopping,
                           model_path=model_path, device=device)
    elif train_test == 'test':
        evaluate(model, valid_data, labels, device=device)
    elif train_test == 'predict':
        predict(model, valid_data, labels, file_path=predict_path, device=device)


def binary_models(train_test, data, data_type, batch_size, epochs, device, model_type, load_model, model_path, optim,
                  lr, bert_model, num_layers_to_delete, weight_method, early_stopping, predict_path, need_other,
                  pretrained_path):
    if data_type != 'facebook':
        if train_test == 'predict':
            train_data, valid = read_bin_data(data, batch_size, need_other=True, all_data=True, data_type=data_type)
        else:
            train_data, valid = read_bin_data(data, batch_size, need_other=need_other, all_data=False, data_type=data_type)
        if data_type == 'hasoc':
            models = {'OFFN': None, 'HATE': None, 'PRFN': None}
        else:
            models = {'ABUSE': None, 'INSULT': None, 'PROFANITY': None}
    else:
        train_data, valid = read_toxic(data, batch_size)
        models = {'TOXIC': None, 'ENGAGING': None, 'FACT': None}
    if load_model or train_test != 'train':
        if pretrained_path is not None:
            models = {cat: torch.load(pretrained_path.format(cat)) for cat in models}
        elif model_path is not None:
            models = {cat: torch.load(model_path.format(cat)) for cat in models}
    else:
        if model_type is None:
            raise Exception("No model type given!")
        for cat in models:
            weights = compute_class_weights(pd.concat(train_data[cat]).binary, weight_method)
            if model_type == "BaseBertModel":
                models[cat] = BaseBertModel(num_class=2, weights=weights, bert_model=bert_model, device=device)
            elif model_type == "MachineLearning":
                models[cat] = MachineLearning(num_class=2, corpus=pd.concat(train_data[cat]).text.tolist(),
                                              weights=weights, device=device)
            else:
                models[cat] = PretrainedBinary(bert_model=bert_model, weights=weights,
                                               num_layers_to_delete=num_layers_to_delete, device=device)
    for cat in models:
        labels = {0: 'OTHER', 1: cat}
        if predict_path is None:
            pp = None
        else:
            pp = predict_path.format(cat)
        run_in_mode(train_test, models[cat], train_data[cat], valid[cat], optim, lr, epochs, model_path.format(cat),
                    labels, early_stopping, pp, device)


def main(train_test, mode, data, data_type, batch_size, epochs, device, model_type, load_model,
         model_path, model_path2, optim, lr, bert_model, num_layers_to_delete, weight_method,
         early_stopping, predict_path, need_other, pretrained_path):
    if mode == 'binary_categories':
        binary_models(train_test, data, data_type, batch_size, epochs, device, model_type, load_model, model_path,
                      optim, lr, bert_model, num_layers_to_delete, weight_method, early_stopping, predict_path,
                      need_other, pretrained_path)
    else:
        drop = (mode == 'offense')
        train_data, valid, label_dict, binary_label_dict = read_data(data, batch_size=batch_size, drop_other=drop,
                                                                     data_type=data_type)

        if mode == 'binary':
            labels = {v: k for (k, v) in binary_label_dict.items()}
            if 'binary' in pd.concat(train_data):
                weights = compute_class_weights(pd.concat(train_data).binary, weight_method)
            else:
                weights = [1, 1]
        else:
            labels = {v: k for (k, v) in label_dict.items()}
            weights = compute_class_weights(pd.concat(train_data).labels, weight_method)

        if load_model or train_test != "train":
            if pretrained_path is not None:
                model = torch.load(pretrained_path)
            elif model_path is not None:
                model = torch.load(model_path)
            elif model_path2 is not None:
                model = torch.load(model_path2)
            else:
                raise Exception("No model path given!")
            if model_path is not None and model_path2 is not None:
                model2 = torch.load(model_path2)
        else:
            if model_type is None:
                raise Exception("No model type given!")
            class_count = {'all': 4, 'offense': 3, 'binary': 2}
            if model_type == "BaseBertModel":
                model = BaseBertModel(num_class=class_count[mode], bert_model=bert_model,
                                      weights=weights, device=device)
            elif model_type == "MachineLearning":
                model = MachineLearning(num_class=class_count[mode], corpus=pd.concat(train_data).text.tolist(),
                                        weights=weights, device=device)
            else:
                if mode == 'offense':
                    model = PretrainedOffense(bert_model=bert_model, num_layers_to_delete=num_layers_to_delete,
                                              device=device)
                elif mode == 'binary':
                    model = PretrainedBinary(bert_model=bert_model, num_layers_to_delete=num_layers_to_delete,
                                             device=device)
                else:
                    model = PretrainedBinary(num_class=4, bert_model=bert_model,
                                             num_layers_to_delete=num_layers_to_delete, device=device)

        if train_test == 'test2':
            try:
                predict_two_models(model, model2, valid, label_dict, binary_label_dict, device=device)
            except NameError:
                raise Exception("No second model path given")
        else:
            run_in_mode(train_test, model, train_data, valid, optim, lr, epochs, model_path, labels, early_stopping,
                        predict_path, device)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--config', '-c')
    argparser.add_argument('--train_test', choices=['train', 'test', 'test2', 'predict'], default='train')
    argparser.add_argument('--pretrained_path')
    argparser.add_argument('--mode', choices=['binary', 'offense', 'all', 'binary_categories'], default='all')
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--bert_model', default='bert-base-german-cased')
    argparser.add_argument('--data_type', choices={'twitter', 'facebook', 'hasoc'}, default='twitter')
    argparser.add_argument('--data_path', nargs='+')
    argparser.add_argument('--epochs', default=4, type=int)
    argparser.add_argument('--model_type', choices=['Pretrained', 'BaseBertModel', 'MachineLearning'])
    argparser.add_argument('--load_model', action='store_true')
    argparser.add_argument('--model_path')
    argparser.add_argument('--model_path2')
    argparser.add_argument('--optimizer', choices=['SGD', 'ASGD', 'Adam', 'Adagrad', 'AdamW'], default='ASGD')
    argparser.add_argument('--learning_rate', '-lr', type=float)
    argparser.add_argument('--num_layers_to_delete', '-d', type=int, default=0)
    argparser.add_argument('--weight_method', choices=['scikit', 'avg', 'log', 'ratio', 'effective', 'none'], default='none')
    argparser.add_argument('--early_stopping', choices=['macro_f1', 'acc', 'micro_f1', 'loss', 'none'], default='none')
    argparser.add_argument('--predict_path', default=None)
    argparser.add_argument('--need_other', action='store_true')
    args = argparser.parse_args()
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    optimizer_dict = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'ASGD': torch.optim.ASGD,
                      'Adagrad': torch.optim.Adagrad, 'AdamW': torch.optim.AdamW}
    args = load_config(args)
    main(train_test=args.train_test, mode=args.mode, batch_size=args.batch_size, data=args.data_path,
         data_type=args.data_type, epochs=args.epochs, device=dev, model_type=args.model_type,
         load_model=args.load_model, model_path=args.model_path, model_path2=args.model_path2, optim=args.optimizer,
         lr=args.learning_rate, bert_model=args.bert_model, num_layers_to_delete=args.num_layers_to_delete,
         weight_method=args.weight_method, early_stopping=args.early_stopping, predict_path=args.predict_path,
         need_other=args.need_other, pretrained_path=args.pretrained_path)
