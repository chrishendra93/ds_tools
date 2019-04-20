DATA_PATH=${PWD%/*}/data/kaggle_cc_fraud
kaggle datasets download mlg-ulb/creditcardfraud/ -p $DATA_PATH
