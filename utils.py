import numpy as np
from matplotlib import pyplot as plt
import csv
import math
from torch.autograd import Variable
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import Dataset
import os
import cv2
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold
import random


def js_div(p_output, q_output, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = p_output.softmax(dim=-1)
        q_output = q_output.softmax(dim=-1)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def caps_loss(y_true, y_pred, x, x_recon):
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = js_div(x_recon, x)

    return L_margin + L_recon

def show_reconstruction(model, test_loader, n_images, args):
    from PIL import Image

    model.eval()
    for x, _ in test_loader:
        x = Variable(x[:min(n_images, x.size(0))].cuda())
        _, x_recon = model(x)
        data = np.concatenate([x.data.cpu(), x_recon.data.cpu()])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.result_dir + "real_and_recon.png")
        print()
        print('Reconstructed images are saved to {}real_and_recon.png'.format(args.result_dir))
        print('-' * 70)
        break

def plot_log(log_file, save_file_path, num, args):
    # load data
    keys = []
    values = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure(figsize=(10,4))
    # fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(121)
    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:
            print(values[:, i])
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.title('Loss')

    fig.add_subplot(122)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    fig.savefig(save_file_path + args.feature + '-' + str(num) + '-log.png')

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

class MyDataset(Dataset):
    def __init__(self, img_names, data_path, transform=None):
        super(Dataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),    
            # transforms.Normalize((0.5), (0.5))
        ])
        self.samples = make_dataset(img_names, data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        label = text2vec(label)
        # label = label.view(1, -1)[0]
        # img = cv2.imread(img_path, 0)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

# labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
labels = ['agenttesla', 'autoit', 'bladabindi', 'delf', 'emotet', 'guloader', 'playtech', 'qbot', 'taskun', 'trickbot', 'ursnif']

def text2vec(label):
    m_zeros = torch.zeros(len(labels))
    m_zeros[labels.index(label)] = 1

    return m_zeros

def vec2text(vec):
    vec = torch.argmax(vec, dim=-1)
    label = labels[vec]
    return label

def make_dataset(img_names, data_path):
    samples = []
    for img_name in img_names:
        img_path = data_path + img_name
        label = img_name.split('.')[0]
        samples.append((img_path, label))
    return samples

def predict(net, inputs):
    net.eval()
    with torch.no_grad():
        outputs, _ = net(inputs.cuda())
        outputs = outputs.view(-1, len(labels))      
    return outputs

def get_all_img(all_data_path, seed, k=5):
    skf = StratifiedKFold(n_splits=k,random_state=seed, shuffle=True)
    img_names = np.array(os.listdir(all_data_path))
    labels = []
    for img_name in img_names:
        label = img_name.split('.')[0]
        labels.append(label)
    labels = np.array(labels)
    return img_names, labels, skf


class ConfusionMatrix(object):

    def __init__(self, args, num):
        self.matrix = np.zeros((len(labels), len(labels)))
        self.args = args
        self.num = num
        self.num_classes = len(labels)

    def update(self, pred, label):
        pred = labels.index(pred)
        # pred = list(pred)
        label = labels.index(label)
        # label = list(label)

        for p, t in zip([pred], [label]):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)
		
		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1 Score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            # Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_Score = round((2 * Precision * Recall) / (Precision + Recall), 5) if Precision + Recall != 0 else 0.

            table.add_row([labels[i], Precision, Recall, F1_Score])
        print(table)
        return str(acc)

    def plot(self):
        fig = plt.figure()
    
        matrix = self.matrix
        # print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix (acc='+self.summary()+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(self.args.result_dir + self.args.feature + '-' + str(self.num) + '-ConfusionMatrix.png', dpi = 200)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True