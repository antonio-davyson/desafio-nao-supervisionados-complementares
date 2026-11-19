# %% [markdown]
# # Desafio 1 — GMM (Gaussian Mixture Model) para clusterização do Iris, com Optuna
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
                             completeness_score, silhouette_score, confusion_matrix)

# %% 1. Importar o dataset
iris = load_iris()
X = iris.data          # Atributos das flores
y = iris.target        # Classes reais (usadas apenas para avaliação)

# %% 2. Exploração
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.shape); print(df.dtypes.value_counts().to_dict()); print(df.describe().round(2)); print(df.isna().sum().sum(), "ausentes")

# %% 3. Pré-processamento: os 4 atributos já estão na mesma unidade (cm); padronizamos para o GMM não ser dominado por atributos de maior variância
Xs = StandardScaler().fit_transform(X)

# %% 4. GMM + otimização de hiperparâmetros com Optuna
# O alvo NÃO entra na otimização (cenário não supervisionado): minimiza-se o BIC.
def objetivo(trial):
    gmm = GaussianMixture(
        n_components=trial.suggest_int("n_components", 1, 8),
        covariance_type=trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"]),
        n_init=trial.suggest_int("n_init", 1, 10),
        max_iter=trial.suggest_int("max_iter", 50, 500),
        reg_covar=trial.suggest_float("reg_covar", 1e-6, 1e-1, log=True),
        random_state=42).fit(Xs)
    return gmm.bic(Xs)

optuna.logging.set_verbosity(optuna.logging.WARNING)
estudo = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
estudo.optimize(objetivo, n_trials=100)
print("Melhores hiperparâmetros:", estudo.best_params, "| BIC =", round(estudo.best_value, 2))

# %% 5. Treinar com os melhores hiperparâmetros e clusterizar
melhor = GaussianMixture(random_state=42, **estudo.best_params).fit(Xs)
rotulos = melhor.predict(Xs); prob = melhor.predict_proba(Xs)

# %% 6. Avaliação com as classes reais
def metricas(r):
    return {"ARI": adjusted_rand_score(y, r), "NMI": normalized_mutual_info_score(y, r),
            "homogeneidade": homogeneity_score(y, r), "completude": completeness_score(y, r),
            "silhueta": silhouette_score(Xs, r) if len(set(r)) > 1 else float("nan")}
print("GMM otimizado (Optuna/BIC):", {k: round(v, 3) for k, v in metricas(rotulos).items()})

gmm3 = GaussianMixture(n_components=3, covariance_type="full", n_init=10, random_state=42).fit(Xs)   # referência: k = nº de espécies
r3 = gmm3.predict(Xs)
print("GMM com 3 componentes (referência):", {k: round(v, 3) for k, v in metricas(r3).items()})
print(pd.crosstab(pd.Series(y, name="espécie").map(dict(enumerate(iris.target_names))), pd.Series(r3, name="cluster (k=3)")))

linhas = []
for k in range(1, 9):
    g = GaussianMixture(k, covariance_type="full", n_init=10, random_state=42).fit(Xs)
    linhas.append({"k": k, "BIC": g.bic(Xs), "AIC": g.aic(Xs), "ARI": adjusted_rand_score(y, g.predict(Xs)) if k > 1 else np.nan})
por_k = pd.DataFrame(linhas).round(2); print(por_k.to_string(index=False))

incerta = (prob.max(axis=1) < 0.9).sum(); print(f"Amostras com probabilidade máxima < 0,9 (modelo otimizado): {incerta} de {len(Xs)}")

# %% 7. Visualização
P = PCA(2).fit_transform(Xs)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].scatter(P[:, 0], P[:, 1], c=y, cmap="viridis", s=25); ax[0].set_title("Espécies reais (PCA)")
ax[1].scatter(P[:, 0], P[:, 1], c=r3, cmap="viridis", s=25); ax[1].set_title("GMM com 3 componentes")
ax[2].scatter(P[:, 0], P[:, 1], c=rotulos, cmap="viridis", s=25); ax[2].set_title(f"GMM otimizado (k={melhor.n_components}, {melhor.covariance_type})")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("gmm_clusters.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(por_k["k"], por_k["BIC"], "o-", label="BIC"); ax[0].plot(por_k["k"], por_k["AIC"], "s-", label="AIC"); ax[0].set_xlabel("componentes"); ax[0].legend()
ax[1].plot([t.number for t in estudo.trials], [t.value for t in estudo.trials], "o", ms=3); ax[1].set_xlabel("trial"); ax[1].set_ylabel("BIC"); ax[1].set_title("Histórico do Optuna")
plt.tight_layout(); plt.savefig("gmm_selecao.png", dpi=120); plt.close()

# %% [markdown]
# ## 8. Documentação dos resultados
# - **Dados:** 150 amostras, 4 atributos numéricos (cm), sem ausentes; atributos padronizados antes do GMM.
# - **Otimização (Optuna, minimizando o BIC, sem usar as classes):** hiperparâmetros buscados: n_components (1–8), covariance_type, n_init, max_iter e reg_covar.
#   O BIC prefere 2 componentes com covariância "full" (BIC ≈ 794,7), contra ≈ 801,5 para 3 componentes. Isso se repete para as outras covariâncias testadas manualmente
#   (full com 2 componentes foi a melhor combinação). O modelo otimizado separa a setosa do resto e junta versicolor + virginica:
#   completude 1,00, mas homogeneidade ≈ 0,58 e ARI ≈ 0,57.
# - **GMM com 3 componentes (referência, k = nº de espécies):** ARI ≈ 0,90, NMI ≈ 0,90 — só 5 amostras de versicolor caem no cluster da virginica.
#   É um resultado melhor que o K-Means com k=3 do outro desafio (ARI ≈ 0,73), porque o GMM com covariância "full" modela grupos elípticos e correlacionados, o que se ajusta melhor às espécies do Iris.
# - **Lição:** assim como com a silhueta no K-Means, um critério interno (BIC) pode preferir menos grupos do que as classes reais. A diferença de BIC entre k=2 e k=3 é pequena (≈ 7 pontos),
#   então 3 componentes é uma escolha defensável; o BIC/AIC não substituem o conhecimento do problema.
# - **Probabilidades:** o GMM dá a probabilidade de cada amostra pertencer a cada grupo (`predict_proba`), o que permite identificar flores de fronteira entre versicolor e virginica — algo que o K-Means não oferece.
# - **Limitações:** com 150 amostras, o ajuste de covariâncias "full" (4×4 por componente) é sensível; o resultado depende da inicialização (por isso n_init > 1).
