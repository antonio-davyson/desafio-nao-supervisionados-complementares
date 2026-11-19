# Algoritmos não supervisionados: tópicos complementares (GMM e LOF)

Desafio Rocketseat com duas partes.

## Desafio 1 — GMM para clusterização do Iris, com Optuna
`desafio1_gmm_iris.py`: importa o Iris, explora, padroniza, otimiza o GMM com Optuna (minimizando o BIC, sem usar as classes), treina, clusteriza, avalia com as espécies reais (ARI, NMI, homogeneidade, completude) e visualiza (`gmm_clusters.png`, `gmm_selecao.png`).

- O BIC prefere 2 componentes (separa setosa do resto): ARI ≈ 0,57.
- Com 3 componentes (nº de espécies): ARI ≈ 0,90, melhor que o K-Means no mesmo dataset.

## Desafio 2 — Detecção de anomalias com LOF no dataset de pulsares
`desafio2_lof_pulsar.py`: carrega `pulsar.csv`, padroniza, ajusta o LOF sem usar `target_class`, marca anomalias, avalia com `target_class` (precisão, recall, F1, AUC) e visualiza (`lof_anomalias.png`).

- O LOF vai mal para separar pulsares: AUC-ROC ≈ 0,56 e F1 ≈ 0,15, pouco acima do acaso. A análise explica por quê e sugere alternativas.

## Como executar
```bash
pip install -r requirements.txt
python desafio1_gmm_iris.py
python desafio2_lof_pulsar.py
```
O código de cada desafio traz a documentação dos resultados no final do arquivo.
