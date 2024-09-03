
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
import pandas as pd

class LogDataExtractor:
    def __init__(self, log_paths):
        """
        extract data from one or multiple log files
        log_paths: paths of all the target logs
        """
        for path in log_paths:
            if not os.path.isfile(path):
                raise ValueError('{} does not exist!'.format(path))

        self.log_paths = log_paths

    def is_target_line(self, line, search_signs):
        if not search_signs:
            return True
        else:
            for item in search_signs:
                if line.startswith(item):
                    return True

        return False

    def process_data(self, path, target_key, line_split=', ', key_split=': ', search_signs=None):
        """
        Given a path to a log process the data in a standard way.
        Extract data based on a target key
        
        path: path to the log
        target_keys: key of target data
        line_split: split to generate (key, value) pairs
        key_split: split to generate key and value
        search_sign: if none search for all the lines
        """
        data = []
        with open(path, 'r') as log:
            lines = log.readlines()
            lines = [l.strip().replace('\n', '') for l in lines]

            for line in lines:
                if self.is_target_line(line, search_signs):
                    info_line = line.split(line_split)
                    for pair in info_line:
                        split_pair = pair.split(key_split)
                        key = split_pair[0]
                        if key == target_key and len(split_pair) > 1:
                            value = split_pair[1]
                            data.append(float(value))
        return data

    def extract_data_by_key(self, target_keys, line_split=', ', key_split=': ', search_signs=None):
        """
        extract data from all the log files, according to target keys
        """
        target_data = []
        for key in target_keys:
            data_dict = {} 
            for path in self.log_paths:
                name = path.split('/')[-1].split('.')[0]
                data = self.process_data(path, key, line_split=line_split, key_split=key_split, search_signs=search_signs)
                data_dict[name] = data
            target_data.append(data_dict)
        return target_data

def plot_features(features, labels, num_classes, epoch, prefix='train', save_dir='results'):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = os.path.join(save_dir, prefix)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    save_name = os.path.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def plot_multilines(data_source, labels, save_dir, xlabel='epoch', ylabel='accuracy',title=None, fig_name='fig.pdf', horizon=[], vertical=[], dash_color=[]):
    """
    Generate figure with multiple lines
    """
    if len(data_source) != len(labels):
        raise Exception('len(data_source) != len(labels)')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    figure = plt.figure(figsize=(6, 8))
    for i in range(len(data_source)):
        plt.plot(data_source[i], label=labels[i])
    plt.legend()
    if len(horizon) > 0:
        for z, h in enumerate(horizon):
            if len(dash_color) != len(horizon):
                plt.axhline(y=h, color='r', linestyle='dashed')
            else:
                plt.axhline(y=h, color=dash_color[z], linestyle='dashed')
    if len(vertical) > 0:
        for z, v in enumerate(vertical):
            if len(dash_color) != len(vertical):
                plt.axvline(x=v, color='r', linestyle='dashed')
            else:
                plt.axvline(x=v, color=dash_color[z], linestyle='dashed')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    figure.savefig(os.path.join(save_dir, fig_name))
    plt.close()

def plot_with_uncertainty(data_source, labels, save_path, xlabel='epoch', ylabel='accuracy',title=None, fig_name='fig.pdf'):
    """
    Generate figure with multiple lines and uncertatinty
    data_source: a 3-d list
    label: legend names
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    matplotlib.rcParams['xtick.major.size'] = 12
    colors = sns.color_palette('husl', len(data_source))
    epochs = list(range(data_source[0].shape[1]))
    for i in range(len(data_source)):
            mean_data = np.mean(data_source[i], axis=0, dtype=np.float64)
            std_data = np.std(data_source[i], axis=0, dtype=np.float64)
            print('{}: data shape: {}, std norm: {}, mean_data norm: {}'.format(labels[i], data_source[i].shape, np.linalg.norm(std_data), np.linalg.norm(mean_data)))
            ax.plot(epochs, mean_data, label=labels[i], color=colors[i])
            ax.fill_between(epochs, mean_data-std_data, mean_data+std_data, alpha=0.3, facecolor=colors[i])
    ax.legend()
    ax.grid(linestyle='--', linewidth=0.5)
    name = '{}.pdf'.format(fig_name)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.savefig(os.path.join(save_path, name))
    plt.close()

def plot_roc(scores, y, save_dir, labels=['1']):
    for i in range(scores[0].shape[1]):
        for j in range(len(scores)):
            pos_scores = scores[j][:, i]
            temp_y = (y == i).astype(int)
            fpr, tpr, _ = metrics.roc_curve(temp_y, pos_scores, pos_label=1)
            auc = metrics.roc_auc_score(temp_y, pos_scores)
            label = labels[j] + '(AUC={:.4f})'.format(auc)
            plt.plot(fpr, tpr, label=label)

        plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), color='black', linestyle='dashed')
        plt.legend()
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('One vs Rest ROC vurve of class: {}'.format(i))
        name = 'ROC_class{}.png'.format(i)
        plt.savefig(os.path.join(save_dir, name))
        plt.close()

def plot_trade_off_bar(data1, data2, index, data1_name, data2_name, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    df = pd.DataFrame({data1_name: data1, data2_name: data2}, index=index)
    ax = df.plot.bar(rot=0, figsize=(6, 8))
    ax.figure.savefig(os.path.join(save_dir, '{}_{}_bar.png'.format(data1_name, data2_name)))

if __name__ == '__main__':
    # index = ['robust_target_0.3', 'robust_target_0.4', 'robust_target_0.5']
    # data1_name = 'train_acc(%)'
    # data2_name = 'train_pgd10_acc(%)'
    # data1 = [89.0, 88.8, 87.9]
    # data2 = [78.0, 77.0, 76.0]
    save_dir = 'plots/ablation-test/minist_lenet'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    values = [91.76, 93.68, 95.86, 96.08, 96.15]
    df = pd.DataFrame({'test PGD40 accuracy (%)': values}, index=['0.3', '0.2', '0.1', '0.08', '0.06'])
    ax = df.plot.bar(rot=0, figsize=(6, 6))
    for p, val in zip(ax.patches, values):
        ax.annotate(val, (p.get_x() * 1.005, p.get_height() * 1.001))
    ax.set_ylim([90, 100])
    ax.set_xlabel("robust target")
    ax.figure.savefig(os.path.join(save_dir, 'lenet_c05_bar_pgd40.png'))
    # plot_trade_off_bar(data1, data2, index, data1_name, data2_name, save_dir)

    