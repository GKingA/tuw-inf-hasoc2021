# tuw-inf-hasoc2021

Here you can find the steps to reproduce our results on [HASOC2021](https://hasocfire.github.io/hasoc/2021/).

## BERT system

### Data normalization

Normalise the list of files; train and test:
```
python3 read_data.py 
                     --to_normalise LIST_OF_FILES
                     --normalised_path LIST_OF_NORMALISED_PATHS_IN_ORDER
                     --language en
```
Concatenate files:
```
python3 read_data.py 
                     --normalised_path LIST_OF_NORMALISED_PATHS
                     --concat CONCATENATED_PATH
```

### Run models
Change the data paths in the config files to direct 
to the concatenated normalised path and the test.json
to direct to the normalised test files.

Important note: if the data_path parameter is a single file,
it will be split into train and validation. 
If you want the model to train on the whole data, 
you have to make it the first element of a list, like you can see
in the configs/English_train_whole_data.json file.
```
python3 run_configs.py 
                     --mode train
                     --configs ./configs 
                     --test_files test.json
```

After the training process finished, the best systems
will give a prediction on the given test files. 
These will be put into the predicted dictionary.
The training process is the same for the categorical subtask.

For the binary subtask, you can run the following script
to get the final result:
```
python3 run_configs.py 
                     --mode result
                     --configs ./configs 
                     --test_files test.json
```

For the fine-grained subtask, you can run:
```
python3 voter.py 
                     --binary [English_train_macro_F1.csv]
                     --abuse [English_train_ABUSE_macro_F1.csv]
                     --insult [English_train_INSULT_macro_F1.csv]
                     --profanity [English_train_PROFANITY_macro_F1.csv]
                     --expected [TEST_FILE]
                     --out [OUTPUT_FILE]
```
If the expected file doesn't contain labels, add the --test argument
to the above command.

## Rule system
