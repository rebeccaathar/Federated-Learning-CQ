# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import os
import csv
import numpy as np
import h5py
import pandas as pd
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from utils.data_utils import add_noise_model, add_noise_parameter , calculate_uplink_delay, calculate_downlik_delay, calculate_noise_weights , calculate_num_clients

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        torch.manual_seed(0)
        random.seed(0)
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.client_selection = args.client_selection
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.current_index = 0

        self.clients = []
        self.selected_clients = []
        self.selected_weights = []
        self.select_clients_index = []
        self.mean_delay_per_round = []
        self.weights_list = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = [] #AUC é uma métrica usada para avaliar a performance de um modelo de classificação binária
        self.rs_train_loss = []
        self.rs_accumulative_delay = []

        self.clients_cq = []
        self.data_rate = []
        self.snr = []
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.round_counter = 0
        self.dlg_gap = args.dlg_gap
        self.num_subchannels = 10
        self.batch_num_per_client = args.batch_num_per_client
        self.client_entropy_selected = []

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.no = []
        self.test_auc = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        weight_list =[]
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_test=True)
        
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data),
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow, 
                            )
            self.clients.append(client)
            


        df = pd.read_csv('/home/rebeca/Federated-Learning-CQ/system/channel/channel_metrics.csv')
        
        idx = [i for i in range(self.num_clients)]
        snr_lim = -7
        for i in idx:
            # Selecionando um índice aleatório
            random_index = random.randint(0, len(df) - 1)
            selected_row = df.iloc[random_index]
            
            snr = selected_row['ebno_db']
            no = selected_row['no']
            data_rate = selected_row['data rate']
            
            self.no.append(no)
            self.snr.append(snr)
            self.data_rate.append(data_rate)
            
            if self.client_selection == "cq" or "cq_entropy":
                print(f'client = {i}  SNR = {snr}')
                if snr >= snr_lim: 
                    weights = calculate_noise_weights(no)
                    weight_list.append(weights)
                    self.select_clients_index.append(i)
                

        self.weights_list = [round(weight, 10) for weight in weight_list]

        exp_weights = np.exp(self.weights_list - np.max(self.weights_list))  # Subtraindo o máximo para estabilidade numérica
        print(f'nao normalizado: {self.weights_list}')
        self.weights_list = (exp_weights / np.sum(exp_weights)).tolist()
        print(f'normalizado: {self.weights_list}')

        #self.weights_list = (np.array(self.weights_list) / norm).tolist()

            
        print(f'index: {self.select_clients_index}')


        # idx = [i for i in range(self.num_clients)]
        # snr_lim = 5
        # self.weights_list = []
        # for i in idx:
        #     snr, no, data_rate , bandwidth = cdl_channel_user(i)
        #     self.no.append(no)
        #     self.snr.append(snr)
        #     self.data_rate.append(data_rate)

        #     if self.client_selection == "cq":
        #         print(f'client = {i}  SNR = {snr}')
        #         if snr >= snr_lim: 
        #             weights = calculate_noise_weights(no)
        #             weight_list.append(weights)
        #             self.select_clients_index.append(i)
                

        # self.weights_list = [round(weight, 8) for weight in weight_list]
        # self.weights_list = np.array(self.weights_list / np.sum(self.weights_list)).tolist()
            
        # print(f'index: {self.select_clients_index}')
        # print(f'weights list = {self.weights_list}')


        if self.client_selection == "rr":
            self.num_groups = self.num_clients // self.num_subchannels
            self.groups = {i: self.clients[i::self.num_groups] for i in range(self.num_groups)}


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        # self.random_join_ratio: É uma flag que determina se o número de clientes a serem selecionados 
        # é aleatório dentro de um intervalo (True) ou fixo (False).
        # self.num_join_clients: Número mínimo de clientes a serem selecionados.
        # self.num_clients: Número total de clientes disponíveis.
        # self.current_num_join_clients: Armazena o número atual de clientes a serem selecionados.

        if self.client_selection == "random":
            if self.random_join_ratio:
                #Número de clientes selecionado aleatoriamente  
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))


        elif self.client_selection == "rr":
            group_id = self.round_counter % self.num_groups
            selected_group = self.groups[group_id]

            self.round_counter += 1
            # Select clients from the chosen group
            selected_clients = list(np.random.choice(selected_group, self.num_subchannels, replace=False))

        elif self.client_selection == "cq":
            if self.random_join_ratio:
                #Número de clientes selecionado aleatoriamente  
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            
                selected_clients = [self.clients[i] for i in self.select_clients_index]
        
        elif self.client_selection == "cq_entropy":
            if self.random_join_ratio:
                #Número de clientes selecionado aleatoriamente  
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            
            if self.round_counter == 0:
                selected_clients = list(np.random.choice(self.clients, self.num_subchannels, replace=False))
            else:
                entropies = np.array([client.client_entropy() for client in self.clients])
                entropies[np.isnan(entropies)] = 0

                for client, entropy in zip(self.clients, entropies):
                    client.entropy = entropy

                combined_metric = entropies * self.weights_list

                    # Ordenar os índices dos clientes por entropia em ordem decrescente
                indices_sorted_by_entropy = np.argsort(combined_metric)[::-1]

                    # pega os top 25% clients
                selected_indices = indices_sorted_by_entropy[:self.num_subchannels]
                print(f'Selected índices: {selected_indices}')
                    
                selected_clients = [self.clients[i] for i in selected_indices]

            self.round_counter += 1


        elif self.client_selection == "entropy":
            if self.random_join_ratio:
                #Número de clientes selecionado aleatoriamente  
                self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                self.current_num_join_clients = self.num_join_clients
            
            if self.round_counter == 0:
                selected_clients = list(np.random.choice(self.clients, self.num_subchannels, replace=False))
            else:
                entropies = np.array([client.client_entropy() for client in self.clients])
                entropies[np.isnan(entropies)] = 0

                for client, entropy in zip(self.clients, entropies):
                    client.entropy = entropy

                combined_metric = entropies

                # Ordenar os índices dos clientes por entropia em ordem decrescente
                indices_sorted_by_entropy = np.argsort(combined_metric)[::-1]

                # pega os top 25% clients
                m = 0.25 * 20
                selected_indices = indices_sorted_by_entropy[:self.num_subchannels]
                print(f'Selected índices: {selected_indices}')
                
                selected_clients = [self.clients[i] for i in selected_indices]

            self.round_counter += 1

        return selected_clients
    
    
    #Enviar o modelo global atualizado do servidor para todos os clientes
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            #start_time = time.time()
            
            # Atualização do modelo do cliente. 
            # Aqui os clientes atualizam seus modelos com os parâmetros do modelo global do servidor
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            #Calcula o tempo gasto na operação de envio e o multiplica por 2
            #client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client.send_time_cost['total_cost'] += calculate_downlik_delay(self.global_model, 10e6) 
            

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
                
        if self.client_selection == "random":
            active_clients = random.sample(self.selected_clients, self.num_subchannels)
        
        if self.client_selection == "cq":
            active_clients = np.random.choice(self.selected_clients, size=self.num_subchannels, 
                                              replace=False, p=self.weights_list)

        if self.client_selection == "rr":
            active_clients = random.sample(self.selected_clients, self.num_subchannels)

        if self.client_selection == "entropy":
            active_clients = random.sample(self.selected_clients, self.num_subchannels)
            #active_clients = self.selected_clients
        
        if self.client_selection == "cq_entropy":
            active_clients = random.sample(self.selected_clients, self.num_subchannels)

            
        self.uploaded_ids = [] 
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_entropy = []
        total_delay_round = 0
        tot_samples = 0
        
        for client in active_clients:
            try:
                # print(client.id)
                # ebno_db, no , data_rate, bandwidth = cdl_channel_user(client.id)
                # print(self.data_rate)
                data_rate = self.data_rate[client.id]
                no = self.no[client.id]
                snr = self.snr[client.id]
                client.receive_time_cost['num_rounds'] += 1
                client.receive_time_cost['total_cost'] += calculate_uplink_delay(client.model, data_rate) 

                #uplink_delay = calculate_uplink_delay(client.model, data_rate)
                mean_training_time =  (client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'])
                mean_downlink_delay = (client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds'])
                mean_uplink_delay = (client.receive_time_cost['total_cost'] / client.receive_time_cost['num_rounds'])
                                         
                
                client_time_cost = mean_training_time + mean_downlink_delay + mean_uplink_delay
                total_delay_round += client_time_cost
                
                print(f'Client {client.id}  SNR = {snr:.1f} data rate = {(data_rate):.2f} Mbs  Train delay = {mean_training_time:.3f}s Downlink delay = {mean_downlink_delay:.3f}s Uplink delay = {mean_uplink_delay:.3f}s client time cost: {client_time_cost:.3f}s')
                #     #   f'Bandwidth = {bandwidth}'

                # client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                #         client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            #Limite de Tempo 
            if client_time_cost <= self.time_threthold: 
                tot_samples += client.train_samples
                "se o modelo chegar antes do tempo estabelecido, ele registra o id do cliente"
                self.uploaded_ids.append(client.id)
                #client.train_samples+=no
                "registra os pesos do modelo do cliente"
                self.uploaded_weights.append(client.train_samples)
                #Adding noise 
                "registra o modelo do cliente"
                add_noise_model(client.model, no)
                self.uploaded_models.append(client.model)
            else:
                print(f'lost client:{client.id}')
        
        self.mean_delay_per_round.append((total_delay_round)/len(active_clients))

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.csv".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
    
    
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + "random_noise2_noniid"
            file_path = os.path.join(result_path, "{}.csv".format(algo))
            print("File path: " + file_path)

            # Salvar os resultados em um arquivo CSV
            with open(file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Escrever cabeçalhos
                csvwriter.writerow(['rs_test_acc', 'rs_train_loss', 'rs_test_auc', 'mean_delay_per_round'])
                # Escrever os dados linha por linha
                for i in range(len(self.rs_test_acc)):
                    print(i)
                    row = [self.rs_test_acc[i], self.rs_train_loss[i],self.test_auc[i], self.mean_delay_per_round[i]]
                    csvwriter.writerow(row)

    

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        #É acumulada somando as acurácias individuais dos clientes e dividindo pelo número total de amostras.
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        self.test_auc.append(test_auc)
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
