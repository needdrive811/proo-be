import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Data
from dataset import DataLoaderX
from config import DefaultConfig
from student_model import student_model
from run import run_epoch
import chunk_analysis as ca

opt = DefaultConfig()

#path_to_train='../data/skill_builder/skill_builder_data_sakt_train.csv'

#path_to_test='../data/skill_builder/skill_builder_data_sakt_test.csv'

#if __name__ == '__main__':
def main(path_to_train,path_to_test,sep=','):
    #sad se u dataset mogu slati datframe ili csv

    train_dataset = Data(path_to_csv=path_to_train,train=True,standard_load=False,sep=sep)
    test_dataset = Data(path_to_csv=path_to_test,train=False,standard_load=False,sep=sep)
    train_loader = DataLoaderX(train_dataset, batch_size=opt.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoaderX(test_dataset, batch_size=opt.batch_size, num_workers=4, pin_memory=True)
    num_skills = train_dataset.skill_num

    m = student_model(num_skills=num_skills, state_size=opt.state_size,
                      num_heads=opt.num_heads, dropout=opt.dropout, infer=False)

    torch.backends.cudnn.benchmark = True
    best_auc = 0
    optimizer = optim.Adam(m.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=opt.lr_decay)
    criterion = nn.BCELoss()
    for epoch in range(opt.max_epoch):
        rmse, auc, r2, acc = run_epoch(m, train_loader, optimizer, scheduler, criterion,
                                           num_skills=num_skills, epoch_id=epoch, is_training=True)
        print('Epoch %d:\nTrain metrics: auc: %.3f, acc: %.3f, rmse: %.3f, r2: %.3f' \
              % (epoch + 1, auc, acc, rmse, r2))
        rmse, auc, r2, acc = run_epoch(m, test_loader, optimizer, scheduler, criterion,
                                       num_skills=num_skills, epoch_id=epoch, is_training=False)
        print('\nTest metrics: auc: %.3f, acc: %.3f, rmse: %.3f, r2: %.3f' \
              % (auc, acc, rmse ,r2))

        if auc > best_auc:
            best_auc = auc
            torch.save(m.state_dict(), 'models/sakt_model_auc_{}.pkl'.format(int(best_auc * 1000)))
