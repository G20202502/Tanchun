# Tanchun: Enhancing Flow-Based Application-Layer HTTP(S) DDoS Detection with Textual Semantics

There're steps about how to apply Tanchun for flow-level HTTP(S) DDoS detection. We use the CIC18 datasets (HOIC, LOIC, GoldenEye and Slowloris in **datasets/**) to illustrate Tanchun's generalization to unseen attacks.

As a demonstration, here we use HOIC as the training set and LOIC as the testing set.

## 1. Prepare for the environment
Install the required python packets:
```bash
conda env create -f environment.yml
conda activate tanchun
```

## 2. Extract concept labels with LLM
Use **llm_labeling_for_batch.py** to extract the concept labels for each IP-pair before training. 

First, set the api key and the path of the training datasets in **llm_labeling_for_batch.py**:
```python
api_key = "your api key"
dataset_list = [
    ("datasets/CIC18-DDoS-HOIC/train_set.csv", "HOIC_train.json"),
]
```

Second, run **llm_labeling_for_batch.py** to extract labels:
```bash
python llm_labeling_for_batch.py
```

Then, the labeling results can be found in **llm_labeled_data/HOIC_train.json**.

## 3. Training and testing
Run **train_test.py** for training and evaluation.

First, set the path of the training json file and the path of the testing CSV file in **train_test.py**:
```python
training_data_json = 'llm_labeled_data/HOIC_train.json'
test_csv = "datasets/CIC18-DDoS-LOIC-HTTP/test_set.csv"
```

Second, run **train_test.py**:
```bash
python train_test.py
```

Now, you can see the detection results in CLI.