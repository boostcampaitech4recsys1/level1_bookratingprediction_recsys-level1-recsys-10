import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel,_MyTransformer
from ._models import rmse, RMSELoss
from sklearn.model_selection import KFold

class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)



    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


############ 내가 만든 모델 ###############################
class aFieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.norm_criterion = RMSELoss()
        self.not_norm_criterion = RMSELoss()
        
        self.train_norm_dataloader = data['train_norm_dataloader'] 
        self.valid_norm_dataloader = data['valid_norm_dataloader']
        self.train_not_norm_dataloader = data['train_not_norm_dataloader']
        self.valid_not_norm_dataloader = data['valid_not_norm_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.norm_model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.not_norm_model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.norm_optimizer = torch.optim.Adam(params=self.norm_model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
        self.not_norm_optimizer = torch.optim.Adam(params=self.not_norm_model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
    
    # def train(self):
    #     for epoch in range(self.epochs):
    #         self.norm_model.train()
    #         total_loss = 0
    #         tk0 = tqdm.tqdm(self.train_norm_dataloader, smoothing=0, mininterval=1.0)
    #         for i, (fields, target) in enumerate(tk0):
    #             fields, target = fields.to(self.device), target.to(self.device)
    #             y = self.norm_model(fields)
    #             loss = self.norm_criterion(y, target.float())
    #             self.norm_model.zero_grad()
    #             loss.backward()
    #             self.norm_optimizer.step()
    #             total_loss += loss.item()
    #             if (i + 1) % self.log_interval == 0:
    #                 tk0.set_postfix(loss=total_loss / self.log_interval)
    #                 total_loss = 0

    #         rmse_score = self.predict_norm_train()
    #         print('epoch:', epoch, 'validation: rmse:', rmse_score)
            
    #     for epoch in range(self.epochs):
    #         self.not_norm_model.train()
    #         total_loss = 0
    #         tk0 = tqdm.tqdm(self.train_not_norm_dataloader, smoothing=0, mininterval=1.0)
    #         for i, (fields, target) in enumerate(tk0):
    #             fields, target = fields.to(self.device), target.to(self.device)
    #             y = self.not_norm_model(fields)
    #             loss = self.not_norm_criterion(y, target.float())
    #             self.not_norm_model.zero_grad()
    #             loss.backward()
    #             self.not_norm_optimizer.step()
    #             total_loss += loss.item()
    #             if (i + 1) % self.log_interval == 0:
    #                 tk0.set_postfix(loss=total_loss / self.log_interval)
    #                 total_loss = 0

    #         rmse_score = self.predict_not_norm_train()
    #         print('epoch:', epoch, 'validation: rmse:', rmse_score)
    
    
    def norm_train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.norm_model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_norm_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.norm_model(fields)
                loss = self.norm_criterion(y, target.float())
                self.norm_model.zero_grad()
                loss.backward()
                self.norm_optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_norm_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_norm_train(self):
        self.norm_model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_norm_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.norm_model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)
    
    
    def not_norm_train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.not_norm_model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_not_norm_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.not_norm_model(fields)
                loss = self.not_norm_criterion(y, target.float())
                self.not_norm_model.zero_grad()
                loss.backward()
                self.not_norm_optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_not_norm_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_not_norm_train(self):
        self.not_norm_model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_not_norm_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.not_norm_model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)
    
    
    def predict(self, test_norm_dataloader, test_not_norm_dataloader):
        self.norm_model.eval()
        self.not_norm_model.eval()
        predicts = list()
        df = pd.DataFrame()
        df1 = pd.DataFrame()
        with torch.no_grad():
            for fields in tqdm.tqdm(test_norm_dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.norm_model(fields)
                fields = fields.detach().cpu().numpy()
                y = y.unsqueeze(1).detach().cpu().numpy()
                df = pd.concat([df,pd.concat([pd.DataFrame(fields),pd.DataFrame(y)], axis=1)])
                # predicts.extend(y.tolist())
                
            for fields in tqdm.tqdm(test_not_norm_dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.not_norm_model(fields)
                fields = fields.detach().cpu().numpy()
                y = y.unsqueeze(1).detach().cpu().numpy()
                df = pd.concat([df,pd.concat([pd.DataFrame(fields),pd.DataFrame(y)], axis=1)])
                # predicts.extend(y.tolist())    

        return df
    
    
class MyTransformer:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.MyT_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _MyTransformer(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)



    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts