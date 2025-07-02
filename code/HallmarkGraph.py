
'''
This file is used for load the saved models, test, prediction and analysis.

Input:
    1. cleaned data file (cleaned_data_out.csv)
    2. graph network file (adjacency_matrix)
    3. target file (cleaned_data_test.xlsx)
    4. saved model file (best_model_path)

Output:
    1. the best balanced_acc model's Confusion matrix and AUROC/AUPRC picture
    2. if Whether_to_predict_hard_samples is TRUE, output hard sample'pred (hard_samples_pred_file)
    3. if Whether_to_calculate_the_shap is TRUE, output all the TRUE samples shap value (every level model shap csv and shap absolute means file)

'''


import pprint

import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import losses, optimizers, regularizers, initializers
from keras.layers import Layer
import re
import os
import shap
import gc
import time
import numpy as np
import pandas as pd
from os.path import join
from scipy.sparse import load_npz, csc_matrix, diags
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=All Information, 1=Warning, 2=Error, 3=Serious Error Only
# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs found: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # Avoid excessive GPU memory usage

# Training on multiple GPUs using MirrordStrategy
strategy = tf.distribute.MirroredStrategy()
# Data file address
data_path = 'data/cleaned_data_out.csv'
graph_path = 'adjacency_matrix/Undirected_'
target_path = 'data/cleaned_data_test.xlsx'

cancer_hallmarks_name_list = ['0_Sustaining Proliferative Signal',
                              '1_Evading Growth Suppressor',
                              '2_Resist Cell Death',
                              '3_Enabling Replicative Immortalit',
                              '4_Inducing Angiogenesis',
                              '5_Activating Invasion and Metasta',
                              '6_Genome Instability and Mutation',
                              '7_Tumor promoting Inflammation',
                              '8_Deregulating Cellular Energetic',
                              '9_Avoiding Immune Destruction']

# Read data
all_data = pd.read_csv(data_path, delimiter='\t', header=0, index_col=0).T  # (6448, 552)
all_data_normalized = (all_data - np.min(all_data)) / (np.max(all_data) - np.min(all_data))

# Read graph adjacency matrix
Adjacency_matrix = [None] * len(cancer_hallmarks_name_list)
Adjacency_matrix_tensor = []
for index, chname in enumerate(cancer_hallmarks_name_list):
    Adj_matrix = load_npz(f'{graph_path}{chname}_matrix.npz')
    Adj = Adj_matrix.astype(np.float32)
    # Adding self loops and degree normalization to adjacency matrices
    A = Adj + csc_matrix(np.eye(Adj.shape[0], dtype=np.float32))
    degrees = np.array(A.sum(axis=1)).flatten()
    D_sqrt_inv = diags(1 / np.sqrt(degrees))
    Adjacency_matrix[index] = D_sqrt_inv @ A @ D_sqrt_inv
    # Convert adjacency matrix into tensor
    # Convert CSR matrix to COO format
    coo_matrix = Adjacency_matrix[index].tocoo()

    # Create SparseTensor(COO format)
    indices = np.column_stack((coo_matrix.row, coo_matrix.col))
    L_tensor = tf.sparse.SparseTensor(indices, coo_matrix.data, coo_matrix.shape)
    Adjacency_matrix_tensor.append(L_tensor)

# Read tags/targets/labels
all_label = pd.read_excel(target_path, sheet_name='A∩C_cleaned', header=0)


# Label processing
def Cleaning_data(sample_data, label, X, test_size):
    # Clean the Xth level label of the data based on the provided X
    filtered = label.set_index('(A∩C)_cleaned')
    True_data = filtered[filtered['check']]        # check
    False_data = filtered[filtered['check'] == False]      # check
    target_X = True_data[f'target_{X}']
    target_counts = target_X.value_counts().sort_index().reset_index()
    print(f'There are {True_data.shape[0]} samples with correct labels. Proceeding with label cleaning below.')

    print(f'Level {X} has a total of {target_counts.shape[0]} labels, which are \n{target_counts}')
    none_label = []
    for index, row in True_data.iterrows():
        if pd.isna(row[f'target_{X}']):
            none_label.append(index)
    print(f'\nAt level {X} , a total of {len(none_label)} samples were cleaned. ')
    all_none_lanel = set(False_data.index) | set(none_label)
    train_sample = set(sample_data.index).difference(all_none_lanel)
    cleaned_data = sample_data[sample_data.index.isin(train_sample)]
    none_label_data = sample_data[sample_data.index.isin(set(False_data.index))]
    print(f'At level {X} , there are {cleaned_data.shape[0]} samples available for training. ')
    cleaned_label = target_X.loc[cleaned_data.index]
    # Map string labels to numerical labels.
    label_to_index = pd.Series(target_counts.index, index=target_counts[f'target_{X}']).to_dict()
    print(f'String Label to Numerical Label Mapping Service：')
    pprint.pprint(label_to_index)
    digital_label = cleaned_label.map(label_to_index)
    # onehot_encoding
    label_I = np.eye(target_counts.shape[0])
    onehot_label = label_I[digital_label]

    X_train, X_test, y_train, y_test = train_test_split(cleaned_data, onehot_label, test_size=test_size,
                                                        stratify=onehot_label,   # Stratified Sampling
                                                        random_state=11)        # 11

    return (X, test_size, cleaned_data, onehot_label, none_label_data, target_counts.shape[0], X_train, X_test, y_train, y_test, label_to_index)


Whether_to_predict_hard_samples = False

Whether_to_calculate_the_shap = False


class BioGCN(Layer):
    def __init__(self, L, K, activation=None, use_bias=False,
            kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
        super(BioGCN, self).__init__(**kwargs)
        self.K = K
        self.L = L  # calc_Laplace_Polynom(L, K)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def get_config(self):
        L_serialized = [{
            'indices': Lk.indices.numpy(),
            'values': Lk.values.numpy(),
            'dense_shape': Lk.dense_shape.numpy()
        } for Lk in self.L]

        config = super().get_config()
        config.update({
            "L": L_serialized,
            "K": self.K,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Restore the serialized dictionary format to SparseTensor.
        L_serialized = config.pop("L")
        L_restored = [
            tf.sparse.SparseTensor(
                indices=Lk['indices'],
                values=Lk['values'],
                dense_shape=Lk['dense_shape']
            ) for Lk in L_serialized
        ]
        return cls(L=L_restored, **config)

    @tf.function
    def call(self, input):
        x = input  # 200, 6448, 1       L = 6448,6448
        _, M, Fin = x.get_shape()  # _, 6448, 1
        N = tf.shape(x)[0]
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        x1 = tf.sparse.sparse_dense_matmul(self.L[0], x0) + x0
        x_con = tf.expand_dims(x1, 0)  # 1 x M x Fin*N
        for k in range(1, self.K):
            x2 = tf.sparse.sparse_dense_matmul(self.L[k], x0) + x0
            x_con = concat(x_con, x2)

        x = tf.reshape(x_con, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N, M, Fin * self.K])  # N*M x Fin*K
        x = tf.nn.relu(x)
        return x

def load_model(file_path):
    custom_objects = {'BioGCN': BioGCN,
                      'GlorotUniform': tf.keras.initializers.glorot_uniform,
                      'relu': tf.nn.relu,
                      'softmax': tf.nn.softmax}
    model = tf.keras.models.load_model(file_path, custom_objects=custom_objects)
    print(f"Model loaded from {file_path}")
    return model

# Load the best model and test on the test set.
# you should find the save_models file to change the best_model_path
best_model_path = 'save_models(0.2)_12_9'
level_max_acc = {}
for filename in os.listdir(best_model_path):
    match = re.match(r'my_BioGCN_net_\((.*?)\)_target_(\d+)_\d+_\((.*?)\)\.h5', filename)
    if match:
        train_test_size = float(match.group(1))
        model_level = int(match.group(2))
        balanced_acc = float(match.group(3))
        if model_level not in level_max_acc or balanced_acc > level_max_acc[model_level][1]:
            level_max_acc[model_level] = (train_test_size, balanced_acc, filename)

best_models = []
for model_level, (train_test_size, acc, filename) in sorted(level_max_acc.items()):
    print(f"for target_{model_level}, Best Balanced Acc: {acc}, best model File Name: {filename}")
    best_models.append({
        'train_test_size': train_test_size,
        'level': model_level,
        'best_balanced_acc': acc,
        'filename': filename
    })

# match = re.search(r'\((.*?)\)', best_model_path)
# best_balanced_acc = match.group(1)

for search_best_model in best_models:
    train_test_size, model_level, level_best_balanced_acc, best_model_filename = search_best_model['train_test_size'], search_best_model['level'], search_best_model['best_balanced_acc'], search_best_model['filename']

    # get cleaned data
    level, test_size, clean_data, clean_label, ver_data, label_number, X_train, X_test, y_train, y_test, label_to_index \
        = Cleaning_data(all_data, all_label, model_level, train_test_size)

    # load the best model
    load_best_model = load_model(join(best_model_path, best_model_filename))
    y_preds = load_best_model.predict(np.expand_dims(X_test, axis=2))

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1))
    balanced_acc = balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1))
    # Calculate precision
    pre_micro = precision_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='micro', zero_division=0)
    pre_macro = precision_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='macro', zero_division=0)
    pre_weighted = precision_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='weighted', zero_division=0)
    # Calculate recall
    recall_micro = recall_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='micro', zero_division=0)
    recall_macro = recall_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='macro', zero_division=0)
    recall_weighted = recall_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='weighted', zero_division=0)
    # Calculate f1_score
    f1_micro = f1_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='micro', zero_division=0)
    f1_macro = f1_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='macro', zero_division=0)
    f1_weighted = f1_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1), average='weighted', zero_division=0)
    print(f"This training is for target_{level}, with {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {ver_data.shape[0]} hard samples, and {label_number} labels in total.")

    print("test Accuraccy: %0.4f, \n"
          "balanced Accuraccy: %0.4f, \n"
          "pre_micro: %0.4f, pre_macro: %0.4f, pre_weighted: %0.4f, \n"
          "recall_micro: %0.4f, recall_macro: %0.4f, recall_weighted: %0.4f, \n"
          "f1_micro: %0.4f, f1_marco: %0.4f, f1_weighted: %0.4f"
          % (acc, balanced_acc, pre_micro, pre_macro, pre_weighted, recall_micro, recall_macro, recall_weighted, f1_micro, f1_macro, f1_weighted))

    # Confusion matrix
    conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1))
    print(f"Confusion matrix: \n{conf_mat}")
    plt.figure(figsize=(8, 6))
    colors = ['white', 'blue', 'red']
    nodes = [0, 5/conf_mat.max(), 1]
    cmap_name = 'custom_blue_red'
    cm = LinearSegmentedColormap.from_list(cmap_name, list(zip(nodes, colors)), N=256)
    cm.set_bad('white')
    mask = conf_mat == 0
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap=cm, mask=mask)
    # sns.heatmap(conf_mat, annot=True, center=5, fmt='d', cmap=cm, vmin=0, vmax=conf_mat.max())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Target_{level} Confusion Matrix')
    plt.show()


    # AUC-ROC
    plt.figure(figsize=(5, 5))
    tprs = []
    pr_aucs = []
    precisions = []
    recalls = []
    base_fpr = np.linspace(0, 1, label_number, endpoint=True)
    # Calculate the ROC and AUC for each class.
    for i in range(label_number):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_preds[:, i])
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot(fpr, tpr, color='Grey')
        tprs.append(np.interp(base_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        # Precision - Recall
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_preds[:, i])
        pr_auc = auc(recall, precision)
        # plt.plot(recall, precision, label=f'Class {i} (AUPRC = {pr_auc:.2f})')
        plt.plot(recall, precision, linestyle='--', color='Grey')
        pr_aucs.append(pr_auc)  # Store AUPRC values for each category
        precisions.append(precision)
        recalls.append(recall)

    # Macro_AUROC
    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    roc_auc_mean = auc(base_fpr, mean_tpr)
    plt.plot(base_fpr, mean_tpr, color='red',label=f'Level-{level}-\nMacro-average ROC curve\n(AUROC = {roc_auc_mean:.4f})')

    # Macro_AUPRC
    all_precisions = np.zeros_like(base_fpr)
    for i in range(len(recalls)):
        interp_prec = np.interp(base_fpr, recalls[i][::-1], precisions[i][::-1])
        all_precisions += interp_prec
    all_precisions /= len(recalls)
    macro_pr_auc = auc(base_fpr[::-1], all_precisions[::-1])
    plt.plot(base_fpr, all_precisions, color='blue',label=f'Level-{level}-\nMacro-average Precision-Recall curve\n(AUPRC = {macro_pr_auc:.4f})')
    # Random guessing line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # Set the boundaries and labels of the chart
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (ROC) / Recall (PR)')
    plt.ylabel('True Positive Rate (ROC) / Precision (PR)')
    plt.title(f'Target_{level} Macro-AUROC and Macro-AUPRC Curves')
    plt.legend(loc="lower right", frameon=False, ncol=1)
    plt.show()

    print("over!")


    # Perform testing and prediction on hard samples.
    if Whether_to_predict_hard_samples:
        hard_samples_pred_file = f"paper_analysis/Hard_samples_({train_test_size})_pred_output(12_10).xlsx"
        if level == 1:
            print(f"\nThere are {ver_data.shape[0]} hard samples in total.")
            Hard_preds = load_best_model.predict(np.expand_dims(ver_data, axis=2))
            maxlevel_mapping = {'T000 CNS': 8,
                                'T001 NEBLA': 2,
                                'T002 MESODM STEMlow': 5,
                                'T003 MESODM STEMhigh': 5,
                                'T004 EWING': 1,
                                'T005 LEUK': 8,
                                'T006 LYMPH': 3,
                                'T007 THCA': 4,
                                'T008 THYM': 3,
                                'T009 PCPG': 5,
                                'T010 GI': 5,
                                'T011 LUAD': 5,
                                'T012 SCC/BLCA': 5,
                                'T013 MELA': 4,
                                'T014 BRCA noBAS': 4,
                                'T015 BRCA BAS': 3,
                                'T016 HEPAC': 4,
                                'T017 PAAD': 5,
                                'T018 ACC': 2,
                                'T019 KICC': 4,
                                'T020 KIPCC': 4,
                                'T021 KICH': 2,
                                'T022 UCEC/CECC': 4,
                                'T023 OV': 3,
                                'T024 PRAD': 4,
                                'T025 TGCT SEM': 1}
            # Export to Excel
            all_labels = all_label.set_index('(A∩C)_cleaned')
            unlabel = np.argmax(Hard_preds, axis=1)
            index_to_label = {value: key for key, value in label_to_index.items()}
            unlabel_to_index = pd.Series(unlabel).map(index_to_label)
            unlabel_prediction_df = pd.DataFrame({
                'Sample Name': ver_data.index,
                'disease': all_labels[all_labels.index.isin(set(ver_data.index))]['disease'].values,
                'Original diagnosis': all_labels[all_labels.index.isin(set(ver_data.index))]['original_diagnosis'].values,  # original_diagnosis
                'Combined short name': all_labels[all_labels.index.isin(set(ver_data.index))]['combined short name'].values,      # combined short name
                'Original diagnosis short name': all_labels[all_labels.index.isin(set(ver_data.index))]['original_diagnosis short name'].values,       # original_diagnosis short name
                'max_level': unlabel_to_index.map(maxlevel_mapping),
                f'Predicted_Label_{level}': unlabel_to_index
            })
            unlabel_prediction_df.to_excel(hard_samples_pred_file, sheet_name=f"hard_samples_prediction({train_test_size})", index=False)
            print(f'The predicted data related to level_1 for hard samples has been written to file：{hard_samples_pred_file}')
        else:
            prediction_1_level = pd.read_excel(hard_samples_pred_file, sheet_name=f"hard_samples_prediction({train_test_size})", header=0)
            ver_data_filtered = ver_data[prediction_1_level.set_index('Sample Name')['max_level'] >= level]
            Hard_preds = load_best_model.predict(np.expand_dims(ver_data_filtered, axis=2))
            unlabel = np.argmax(Hard_preds, axis=1)
            index_to_label = {value: key for key, value in label_to_index.items()}
            unlabel_to_index = pd.Series(unlabel).map(index_to_label)
            unlabel_prediction_df = pd.DataFrame({
                'Sample Name': ver_data_filtered.index,
                f'Predicted_Label_{level}': unlabel_to_index
            })
            if f'Predicted_Label_{level}' in prediction_1_level.columns:
                prediction_1_level = prediction_1_level.drop(columns=[f'Predicted_Label_{level}'])

            updated_prediction_df = pd.merge(prediction_1_level, unlabel_prediction_df, on='Sample Name', how='left')
            # updated_prediction_df.to_excel(hard_samples_pred_file, sheet_name=f"unlabel_prediction_level_{level}", index=False)
            with pd.ExcelWriter(hard_samples_pred_file, engine='openpyxl', mode='a') as writer:  # mode='a' means "append mode"
                if f"hard_samples_prediction({train_test_size})" in writer.book.sheetnames:
                    del writer.book[f"hard_samples_prediction({train_test_size})"]
                updated_prediction_df.to_excel(writer, sheet_name=f"hard_samples_prediction({train_test_size})", index=False)
            print(f'The predicted data related to level_{level} for hard samples has been written to file：{hard_samples_pred_file}')



    # Calculate the SHAP value
    if Whether_to_calculate_the_shap:

        import torch
        import random
        tf.keras.backend.clear_session()
        torch.cuda.empty_cache()

        # SHAP explainer
        if level == 8:       # In the training of the 8th label, there are only 94 training data for the 6-4 ratio
            background_samples = 94
        else:
            background_samples = 100
        random.seed(12)
        background_indices = np.random.choice(len(X_train), background_samples, replace=False)
        background = np.expand_dims(X_train.iloc[background_indices], 2)


        explainer_lode = shap.DeepExplainer(load_best_model, background)

        batch_size = 400

        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["SparseTensorDenseMatMul"] = shap.explainers._deep.deep_tf.passthrough

        print("Start calculating the SHAP values for all data: \n")

        data_combind = pd.concat([X_train,X_test],axis=0)       # data for calculate shap
        label_combind = np.concatenate((y_train, y_test), axis=0)   # label for calculate shap


        train_shap = np.expand_dims(data_combind, 2)
        num_batches = int(np.ceil(len(train_shap) / batch_size))
        batches = np.array_split(train_shap, num_batches)
        print(f"Start calculating the SHAP values for {data_combind.shape[0]} samples, totaling {num_batches} blocks: \n")
        shap_values_train_list = []
        start_index = 0
        for batch_index, batch in enumerate(batches, start=0):
            shap_values_train = explainer_lode.shap_values(batch)

            end_index = start_index + batch.shape[0]
            batch_labels = label_combind[start_index: end_index]
            select_labels = np.argmax(batch_labels, axis=1)
            shap_values_train_3 = np.squeeze(shap_values_train, axis=2)     # (xxx, 6448, label_number)

            for i in range(shap_values_train_3.shape[0]):
                batch_shap_values = shap_values_train_3[i,:,select_labels[i]]
                shap_values_train_list.append(batch_shap_values)
            start_index = end_index
            print(f"For level_{level}, calculated { batch_index + 1 } / {num_batches} batches completed")

        shap_values_all = np.vstack(shap_values_train_list)         # (xxx, 6448)
        filtered_shap_values_all = np.array(shap_values_all)                              # (xxx, 6448)
        np.savetxt(f'paper_analysis/level_{level}_load_best_model_({level_best_balanced_acc})_shap.csv', filtered_shap_values_all, delimiter=',', fmt='%f')

        shap_absolute_means = np.mean(np.abs(filtered_shap_values_all), axis=0)

        time_name_name = f'level_{level}_{level_best_balanced_acc}_100'
        shap_absolute_means_df = pd.DataFrame(shap_absolute_means, columns=[time_name_name])
        shap_absolute_means_df.index = X_train.columns
        shap_values_path = f'paper_analysis/sample_shap_values_level_(11_17).xlsx'

        # Export to Excel
        if level == 1:
            mode = 'w'
        else:
            mode = 'a'
        with pd.ExcelWriter(shap_values_path, engine='openpyxl', mode=mode) as writer:  # mode='a' means "append mode"
            if time_name_name in writer.book.sheetnames:
                del writer.book[time_name_name]
            shap_absolute_means_df.to_excel(writer, sheet_name=time_name_name, index=True)
        print(f'The data related to {time_name_name} has been written to the file: {shap_values_path}. Please check it out.')




    # Clean the GPU memary
    del load_best_model
    del clean_data, clean_label, ver_data, label_number, X_train, X_test, y_train, y_test
    gc.collect()

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
