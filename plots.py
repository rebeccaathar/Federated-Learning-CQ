import pandas as pd
import matplotlib.pyplot as plt


df_entropy = pd.read_csv('./results/FashionMNIST_FedAvg_test_entropy_noiseless_noniid_simple.csv')


accuracies_fedavg_RS = df_entropy['rs_test_acc']


rounds = range(1, len(accuracies_fedavg_RS) + 1)  # Números dos rounds
print(rounds)

plt.figure(figsize=(10, 6))
plt.plot(rounds, accuracies_fedavg_RS, marker='o', linestyle='-', color='b', label='Entropy noiseless')
# plt.plot(rounds, accuracies_fedprox_RS, marker='o', linestyle='-', color='g', label='FedProx')


#plt.title('Comparação de Acurácia por Round')
plt.xlabel('Communication round', fontsize = 20)
plt.ylabel('Test accuracy', fontsize = 20)
plt.grid(True)
plt.legend()
plt.tight_layout()

#Salvar a figura em um arquivo
plt.savefig('./results/FashionMNIST_FedAvg_Entropy_noiseless_noniid_simples_acc.png')

