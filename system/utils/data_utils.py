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

import numpy as np
import os
import torch
import sys
import yaml
from typing import List



def calculate_noise_weights(noise_level):
    #noise_level = "{:.3f}".format(noise_level)
    weight = 1.0/(1.0 + float(noise_level)) # # Quanto menor o ruído, maior o peso
    #print(f'noise: {noise_level}  weight: {weight}')
    return weight


def calculate_num_clients(min, max):
    return np.random.randint(min, max + 1)


def add_noise_parameter(parameter, noise):
    
    print(f'numero de samples: {parameter}')
    parameter= np.random.randn(parameter) * noise
    print(f'samples dois do ruído: {parameter}')


def add_noise_model(model, noise):
    for param in model.parameters():
        #Gera um tensor com a mesma forma que param.data, 
        #contendo valores aleatórios de uma distribuição normal padrão (média 0, desvio padrão 1).
        #Ajusta a escala do ruído multiplicando pelo desvio padrão noise_std.
        param.data += torch.randn_like(param.data) * noise



# def add_noise(model, noise_std):
#     for param in model.parameters():
#         noise = torch.randn_like(param.data) * noise_std
#         # Arredonda o tensor de ruído para 3 casas decimais
#         noise = torch.round(noise * 1000) / 1000
#         # Adiciona o ruído aos parâmetros do modelo
#         param.data.add_(noise)

def get_data_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.element_size() * param.numel()
    return total_size 


def calculate_downlik_delay(server_parameters, server_data_rate):
    parameters_size = get_data_size(server_parameters)
    delay_downlink = parameters_size/server_data_rate
    return delay_downlink

def calculate_uplink_delay(client_model, client_data_rate):
    parameters_model = get_data_size(client_model) #bytes
    #print(f'parameter size: {parameters_model}')
    delay_uplink = parameters_model*8/client_data_rate #bit/bits/s 
    return delay_uplink

def calculate_data_rate(bandwidth, snr_db, num_bits_per_symbol, coderate):
    """
    Calcula a capacidade do canal usando a fórmula de Shannon-Hartley.

    Parâmetros:
    bandwidth (Hz): Largura de banda do canal em Hz
    snr_db (dB): Relação Sinal-Ruído em decibéis
    spectral_efficiency (bit/s/Hz): Eficiência espectral

    Retorna:
    capacidade (bps): Capacidade do canal em bits por segundo
    """

    spectral_efficiency = num_bits_per_symbol * coderate

    # Converter SNR de dB para valor linear
    snr_linear = 10 ** (snr_db / 10)
    
    # Calcular a capacidade do canal usando a fórmula de Shannon-Hartley
    capacity = bandwidth * np.log2(1 + snr_linear)
    
    # Ajustar a capacidade pela eficiência espectral
    data_rate = capacity * spectral_efficiency
    
    return data_rate

def calculate_upload_time(data_size, data_rate):
    return data_size/data_rate

#---------------------------------------------------------
def create_number_list(x: int) -> List[int]:
    """Cria uma lista de números de 1 a x."""
    return list(range( x ))


def read_data(dataset, idx, is_train=True, is_val=False,  is_test=False):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data
    
    if is_val:
        val_data_dir = os.path.join('../dataset', dataset, 'val/')

        val_file = val_data_dir + str(idx) + '.npz'
        with open(val_file, 'rb') as f:
            val_data = np.load(f, allow_pickle=True)['data'].tolist()

        return val_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True, is_val=False, is_test=False):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train, is_val, is_test)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    
    if is_val:
        val_data = read_data(dataset, idx, is_val)
        X_val = torch.Tensor(val_data['x']).type(torch.float32)
        y_val = torch.Tensor(val_data['y']).type(torch.int64)
        val_data = [(x, y) for x, y in zip(X_val, y_val)]
        return val_data
    
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=False, is_val=False, is_test=False):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    
    if is_val:
        val_data = read_data(dataset, idx, is_val)
        X_val, X_val_lens = list(zip(*val_data['x']))
        y_val = val_data['y']

        X_val = torch.Tensor(X_val).type(torch.int64)
        X_val_lens = torch.Tensor(X_val_lens).type(torch.int64)
        y_val = torch.Tensor(val_data['y']).type(torch.int64)

        val_data = [((x, lens), y) for x, lens, y in zip(X_val, X_val_lens, y_val)]
        return val_data

    else:
        test_data = read_data(dataset, idx, is_test)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

