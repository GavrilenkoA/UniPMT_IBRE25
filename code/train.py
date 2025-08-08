import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
import torch.nn as nn

from dataloader import Dataset
from model import MolGNN
import config.config as config


class Trainer:
    def __init__(self, path, data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.graph = data.graph_dataset.graph_data
        self.p_num = self.graph['p'].x.shape[0]
        self.t_num = self.graph['t'].x.shape[0]
        self.m_num = self.graph['m'].x.shape[0]
        self.pt_m_dict = data.graph_dataset.pt_m_dict
        meta = self.graph.metadata()

        pm_dataset = data.pm_dataset
        pt_dataset = data.pt_dataset
        pmt_dataset = data.pmt_dataset

        pm_data_len = len(pm_dataset)
        pt_data_len = len(pt_dataset)
        pmt_data_len = len(pmt_dataset)
        n_batch = max(1, pmt_data_len // config.batch_size)
        pm_batch = max(1, pm_data_len // n_batch)
        pt_batch = max(1, pt_data_len // n_batch)

        self.pm_loader = DataLoader(pm_dataset, batch_size=pm_batch, shuffle=True, num_workers=4)
        self.pt_loader = DataLoader(pt_dataset, batch_size=pt_batch, shuffle=True, num_workers=4)
        self.pmt_loader = DataLoader(pmt_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

        self.model = MolGNN(path, self.p_num, self.t_num, self.m_num, config.hidden_size, meta, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.reg_lambda)
        self.bce = nn.BCELoss()

    def pm_loss(self, pm_batch):
        p_gnn_emb = self.p_out_emb[pm_batch[:, 0]]
        m_gnn_emb = self.m_out_emb[pm_batch[:, 1]]
        labels = pm_batch[:, 2].float().to(self.device).unsqueeze(-1)
        pm_pred = self.model.pm_pred(p_gnn_emb, m_gnn_emb)
        return self.bce(pm_pred, labels)

    def pt_loss(self, pt_batch):
        p_indices, t_indices, m_indices, leaveout = [], [], [], []
        for i, pt_sample in enumerate(pt_batch):
            tup = tuple(pt_sample[:2].tolist())
            if pt_sample[2] == 1 and tup in self.pt_m_dict:
                p_indices.append(pt_sample[0])
                t_indices.append(pt_sample[1])
                m_indices.append(np.random.choice(list(self.pt_m_dict[tup])))
                leaveout.append(i)
            else:
                p_indices += [pt_sample[0]] * self.m_num
                t_indices += [pt_sample[1]] * self.m_num
                m_indices += list(range(self.m_num))
                leaveout += [i] * self.m_num

        p_indices = torch.tensor(p_indices).to(self.device)
        t_indices = torch.tensor(t_indices).to(self.device)
        m_indices = torch.tensor(m_indices).to(self.device)
        p_gnn_emb = self.p_out_emb[p_indices]
        t_gnn_emb = self.t_out_emb[t_indices]
        m_gnn_emb = self.m_out_emb[m_indices]

        pt_pred_flat = self.model.pt_pred(p_gnn_emb, t_gnn_emb, m_gnn_emb).squeeze()
        pt_pred = scatter_mean(pt_pred_flat, torch.tensor(leaveout).to(self.device), dim=0)
        labels = pt_batch[:, 2].float().to(self.device)
        return self.bce(pt_pred.unsqueeze(-1), labels.unsqueeze(-1))

    def pmt_loss(self, pmt_batch):
        pmt_pos, pmt_negs, _ = self.model.pmt_learn(self.p_out_emb, self.m_out_emb, self.t_out_emb, pmt_batch)
        pos = pmt_pos.squeeze(-1)
        neg = torch.stack(pmt_negs, dim=1).squeeze(-1)
        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss = nn.CrossEntropyLoss()(logits / config.tau, labels)
        return loss

    def train(self):
        for epoch in range(config.epoch_num):
            self.model.train()
            for pm_batch, pt_batch, pmt_batch in zip(self.pm_loader, self.pt_loader, self.pmt_loader):
                self.optimizer.zero_grad()
                gnn_out = self.model.gnn_learn(self.graph)
                self.p_out_emb = gnn_out['p']
                self.m_out_emb = gnn_out['m']
                self.t_out_emb = gnn_out['t']

                loss_pm = self.pm_loss(pm_batch)
                loss_pt = self.pt_loss(pt_batch)
                loss_pmt = self.pmt_loss(pmt_batch)
                loss = config.pm_weight * loss_pm + config.pt_weight * loss_pt + config.pmt_weight * loss_pmt
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: L_pm={loss_pm.item():.4f} L_pt={loss_pt.item():.4f} L_pmt={loss_pmt.item():.4f}")
        torch.save(self.model.state_dict(), config.model_path)


def main():
    path = '../data/{}/meta'.format(config.data_folder)
    print(f"processing {config.data_folder}")
    if config.regenerate_graphdata:
        import shutil
        if os.path.exists(path + "/processed"):
            print("remove path/processed folder")
            shutil.rmtree(path + "/processed")
    data = Dataset(path)

    trainer = Trainer(path, data)
    print("Start Training...")
    trainer.train()


if __name__ == '__main__':
    main()
