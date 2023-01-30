import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix


best_dl_path = 'test_MLP.csv'
best_benchmark_path = 'test_probit.csv'

dl_results = pd.read_csv(
    best_dl_path, encoding="ISO-8859-1", sep=',', decimal='.', index_col=0)
benchmark_results = pd.read_csv(
    best_benchmark_path, encoding="ISO-8859-1", sep=',', decimal='.', index_col=0)

true = np.sign(benchmark_results['true'])
dl_pred = np.sign(dl_results['pred'])
bench_pred = np.where(benchmark_results['pred'] > 0.3, 1, 0)
heuristic_pred = [1]*len(dl_pred)

dl_pred = np.where(dl_pred == -1, 0, dl_pred)

dl_correct = true == dl_pred
bench_correct = true == bench_pred
heuristic_correct = true == heuristic_pred

print("Heuristic vs DL")
cont_table = np.array([[0, 0], [0, 0]])

cont_table[0][0] = sum(
    [i and j for i, j in zip(dl_correct, heuristic_correct)])
cont_table[0][1] = sum(
    [i and not j for i, j in zip(dl_correct, heuristic_correct)])
cont_table[1][0] = sum(
    [not i and j for i, j in zip(dl_correct, heuristic_correct)])
cont_table[1][1] = sum(
    [not i and not j for i, j in zip(dl_correct, heuristic_correct)])

print(cont_table)

print('Chi square score: {}'.format((np.abs(
    cont_table[0][1]-cont_table[1][0])-1)**2 / (cont_table[0][1]+cont_table[1][0])))
print('P-value: {}'.format(mcnemar(cont_table).pvalue))


print("\nBenchmark vs DL")
cont_table = np.array([[0, 0], [0, 0]])

cont_table[0][0] = sum([i and j for i, j in zip(dl_correct, bench_correct)])
cont_table[0][1] = sum(
    [i and not j for i, j in zip(dl_correct, bench_correct)])
cont_table[1][0] = sum(
    [not i and j for i, j in zip(dl_correct, bench_correct)])
cont_table[1][1] = sum(
    [not i and not j for i, j in zip(dl_correct, bench_correct)])

print(cont_table)

print('Chi square score: {}'.format((np.abs(
    cont_table[0][1]-cont_table[1][0])-1)**2 / (cont_table[0][1]+cont_table[1][0])))
print('P-value: {}'.format(mcnemar(cont_table).pvalue))

print('Return')
print('DL')

return_dl = np.where(dl_correct == True, np.abs(
    dl_results['true']), -np.abs(dl_results['true']))
return_benchmark = np.where(bench_correct == True, np.abs(
    dl_results['true']), -np.abs(dl_results['true']))
return_naive = np.where(heuristic_correct == True, np.abs(
    dl_results['true']), -np.abs(dl_results['true']))

print(np.mean(return_dl))
print(np.mean(return_benchmark))
print(np.mean(return_naive))
