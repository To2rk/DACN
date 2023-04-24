import torch
import argparse
from time import time
import csv
from torch.optim import lr_scheduler, AdamW
from torch.autograd import Variable
from models import CapsuleNet
from utils import MyDataset, plot_log, caps_loss, setup_seed, get_all_img, ConfusionMatrix, predict, vec2text
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import numpy as np

def test(model, valid_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in valid_loader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon).item() * x.size(0)
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)
        for i, j in zip(y_pred, y):
            if torch.equal(i, j):
                correct += 1

    test_loss /= len(valid_loader.dataset)

    if args.epochs == epoch + 1:

        confusion = ConfusionMatrix(args, num)

        for img, target in valid_loader:
            pred = predict(model, img)
            for e in range(len(pred)):
                tag = vec2text(target[e])     
                pre = vec2text(pred[e])     

                confusion.update(pre, tag)                        

        confusion.plot()
        confusion.summary()

    return test_loss, correct / len(valid_loader.dataset)


def train(model, train_loader, valid_loader, args, num):
    print('Begin Training' + '-'*70)

    logfile = open(args.result_dir + args.feature + '-' + str(num) + '-log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_loss = 100
    for epoch in range(args.epochs):
        model.train()
        ti = time()
        training_loss = 0.0
        train_acc = 0
        correct = 0
        for x, y in train_loader:
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon)

            y_pred = torch.argmax(y_pred, dim=1)
            y = torch.argmax(y, dim=1)
            for i, j in zip(y_pred, y):
                if torch.equal(i, j):
                    correct += 1
            
            loss.backward()
            training_loss += loss.item() * x.size(0) 
            optimizer.step()

        train_loss = training_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        val_loss, val_acc = test(model, valid_loader, epoch)
        scheduler.step()
        logwriter.writerow(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))

        print('Epoch: {:03d}'.format(epoch),
            'train_loss: {:.5f}'.format(train_loss),
            'train_acc: {:.3f}%'.format(train_acc * 100),
            'val_loss: {:.5f}'.format(val_loss),
            'val_acc: {:.3f}%'.format(val_acc * 100),
            'time: {:0.2f}s'.format(time() - ti))

        if args.epochs == epoch + 1:
            val_acc = round(val_acc, 5)
            torch.save(model.state_dict(), args.models_dir + args.feature + '-' + str(num) + '-trained.pkl')
            print("val_acc increased to %.3f%%" % (val_acc*100))

    logfile.close()
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model, val_acc

def ArgumentParser():
    parser = argparse.ArgumentParser(description="Capsule Network on Malware Dataset.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--classes', default=11, type=int)
    parser.add_argument('--input_size', default=[3, 28, 28], type=list)
    parser.add_argument('--feature', default='api+dll+reg', type=str)
    parser.add_argument('--lr', default=0.00032, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.99, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=3, type=int, help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_pixels', default=2, type=int, help="Number of pixels to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data/DACN/cross-validation/',help="Directory of data.")
    parser.add_argument('--result_dir', default='./results/')
    parser.add_argument('--models_dir', default='./models/')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = ArgumentParser()

    setup_seed(args.seed)

    table = PrettyTable()
    table.field_names = ["Epochs", "Batch Size", "lr", "Cross Validation", "1", "2", "3", "4", "5", "Average"]

    # load all data
    k = 5 
    num = 0
    acc = []
    X , y, skf = get_all_img(args.data_dir + args.feature + '/', args.seed, k)
    for train_index, valid_index in skf.split(X, y):
        num = num + 1
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_test = y[train_index], y[valid_index]

        train_dataset = MyDataset(X_train, args.data_dir + args.feature + '/', True)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

        valid_dataset = MyDataset(X_valid, args.data_dir + args.feature + '/', True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

        model = CapsuleNet(input_size=args.input_size, classes=args.classes, routings=args.routings)
        model.cuda()

        _, val_acc = train(model, train_data_loader, valid_data_loader, args, num)
        acc.append(val_acc)
        plot_log(args.result_dir + args.feature + '-' + str(num) + '-log.csv', args.result_dir, num, args)

    Average = round(np.mean(acc), 5)
    table.add_row([args.epochs, args.batch_size, args.lr, 'acc', acc[0], acc[1], acc[2], acc[3], acc[4], Average])

    print(table)
    with open(args.result_dir + args.feature + '-' + str(args.epochs) + '-' + str(args.batch_size) + '-'  + str(args.lr) + '-'  + 'ACC-Cross_Validation.txt', 'w') as outputfile:
        outputfile.write(str(table))
