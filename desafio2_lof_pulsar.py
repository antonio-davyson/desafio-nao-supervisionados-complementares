# %% [markdown]
# # Desafio 2 — Detecção de anomalias com LOF (Local Outlier Factor) no dataset de pulsares
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

# %% 1. Carregar e explorar
df = pd.read_csv("pulsar.csv")
print(df.shape); print(df.dtypes.value_counts().to_dict()); print(df.isna().sum().sum(), "ausentes,", df.duplicated().sum(), "duplicatas")
print(df["target_class"].value_counts(normalize=True).round(4))
print(df.describe().T[["mean", "std", "min", "max"]].round(2))

# %% 2. Pré-processamento: o alvo fica de fora (cenário não supervisionado) e os atributos são padronizados (LOF usa distâncias)
y = df["target_class"].values
X = StandardScaler().fit_transform(df.drop(columns="target_class"))

# %% 3. LOF + marcação das anomalias
def rodar_lof(n_neighbors, contamination):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = lof.fit_predict(X)                  # -1 = anomalia, 1 = normal
    return (pred == -1).astype(int), -lof.negative_outlier_factor_   # (1 = anomalia, escore: maior = mais anômalo)

def avaliar(pred, escore):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"anomalias": int(pred.sum()), "precisão": precision_score(y, pred), "recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "AUC-ROC": roc_auc_score(y, escore), "AUC-PR": average_precision_score(y, escore),
            "VP": tp, "FP": fp, "FN": fn}

# %% 4. Avaliação com target_class (pulsar = 1 tratado como "anomalia"). A contaminação é o parâmetro mais sensível.
prev = y.mean()
res = []
for k in [10, 20, 50, 100]:
    for c in [0.02, 0.05, round(prev, 3), 0.15]:
        pred, esc = rodar_lof(k, c)
        res.append({"n_neighbors": k, "contamination": c, **avaliar(pred, esc)})
res = pd.DataFrame(res); print(res.round(3).to_string(index=False))
print("\nProporção real de pulsares (base de comparação):", round(prev, 4))

# %% 5. Configuração principal: n_neighbors=20, contamination = proporção de pulsares
pred, esc = rodar_lof(20, round(prev, 3))
m = avaliar(pred, esc)
print("LOF (n_neighbors=20, contamination=%.3f):" % round(prev, 3), {k: round(v, 3) for k, v in m.items()})
print(pd.crosstab(pd.Series(y, name="real (1=pulsar)"), pd.Series(pred, name="LOF (1=anomalia)")))
print("Sorteio aleatório de referência: precisão ≈", round(prev, 3), "| AUC-ROC ≈ 0,5")

# Análise por tipo: que perfil as anomalias têm? (médias padronizadas dos atributos)
Xd = pd.DataFrame(X, columns=df.columns[:-1])
print(Xd.groupby(pred).mean().T.round(2).rename(columns={0: "normal", 1: "anomalia"}))

# %% 6. Visualização (projeção PCA 2D)
P = PCA(2).fit_transform(X)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(P[y == 0, 0], P[y == 0, 1], s=3, alpha=.3, label="não-pulsar"); ax[0].scatter(P[y == 1, 0], P[y == 1, 1], s=5, c="red", label="pulsar (real)")
ax[0].set_title("Classes reais"); ax[0].legend(markerscale=3)
ax[1].scatter(P[pred == 0, 0], P[pred == 0, 1], s=3, alpha=.3, label="normal"); ax[1].scatter(P[pred == 1, 0], P[pred == 1, 1], s=5, c="red", label="anomalia (LOF)")
ax[1].set_title("Anomalias detectadas pelo LOF"); ax[1].legend(markerscale=3)
ax[2].hist(esc[y == 0], bins=80, alpha=.6, range=(0.8, 4), label="não-pulsar", density=True); ax[2].hist(esc[y == 1], bins=80, alpha=.6, range=(0.8, 4), label="pulsar", density=True)
ax[2].set_xlabel("escore LOF (maior = mais anômalo)"); ax[2].set_title("Distribuição do escore LOF"); ax[2].legend()
for a in ax[:2]: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("lof_anomalias.png", dpi=120); plt.close()

# %% [markdown]
# ## 7. Documentação dos resultados
# - **Dados:** 17.898 observações, 8 atributos numéricos em escalas muito diferentes (padronizados), sem ausentes nem duplicatas. 9,16% das observações são pulsares (1.639).
# - **Cenário:** o LOF foi ajustado sem usar `target_class`, e as anomalias (rótulo -1) foram comparadas depois com o alvo, tratando pulsar como "anomalia".
# - **Resultado principal (n_neighbors=20, contamination=0,092):** 1.647 anomalias marcadas, das quais só 248 são pulsares reais (precisão ≈ 15%, recall ≈ 15%, F1 ≈ 0,15).
#   AUC-ROC ≈ 0,56 e AUC-PR ≈ 0,12, apenas um pouco acima de um sorteio aleatório (precisão esperada ≈ 9%, AUC-ROC 0,5).
# - **Sensibilidade aos parâmetros:** variar n_neighbors (10 a 100) e contamination (0,02 a 0,15) melhora pouco: o melhor AUC-ROC foi ≈ 0,58 (n_neighbors 50–100) e o maior F1 ≈ 0,20 com contamination 0,15,
#   às custas de muitos falsos positivos (≈ 2.260 não-pulsares marcados).
# - **Por que o LOF vai mal aqui:** o LOF procura pontos em regiões de densidade local menor que a vizinhança. Os pulsares são a classe minoritária, mas formam grupos
#   com densidade própria e muitos deles estão misturados à região dos não-pulsares; além disso, as anomalias que o LOF encontra (leve viés para valores altos de média e desvio da curva DM-SNR)
#   são pontos extremos de qualquer classe (assimetria e curtose com cauda muito longa), e não pulsares em particular.
# - **Conclusão:** como detector não supervisionado, o LOF não é adequado para separar pulsares de não-pulsares neste dataset.
#   Com rótulos disponíveis, o modelo supervisionado (regressão logística, AUC ≈ 0,97) é muito melhor; o LOF faria mais sentido para achar observações *incomuns* (ruído, erros de medição) e não uma classe específica.
# - **Melhorias possíveis:** transformar atributos com cauda longa (log), selecionar atributos mais informativos, testar Isolation Forest, One-Class SVM ou Autoencoder, e usar o LOF como filtro de qualidade dos dados.
