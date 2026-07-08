# ====================================================================
# MULTIVARIATE ANALYSIS TOOL — WEB APP (Streamlit)
# All analysis functions are identical to the Colab version.
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, silhouette_score, silhouette_samples)
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.decomposition import PCA, FactorAnalysis as SKLearnFA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import semopy
import io
import zipfile
import base64
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
#  Global settings
# --------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# --------------------------------------------------------------------
#  ANALYSIS FUNCTIONS (copied verbatim from the Colab version)
# --------------------------------------------------------------------

# 1. CORRELATION
def correlation_analysis(data, vars_list, corr_method='pearson',
                         alpha=0.05, threshold=0.5):
    df_corr = data[vars_list]
    if corr_method == 'pearson':
        corr_matrix = df_corr.corr(method='pearson')
        p_matrix    = df_corr.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
    elif corr_method == 'spearman':
        corr_matrix = df_corr.corr(method='spearman')
        p_matrix    = df_corr.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
    else:
        corr_matrix = df_corr.corr(method='kendall')
        p_matrix    = df_corr.corr(method=lambda x, y: stats.kendalltau(x, y)[1])

    sig_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if p_matrix.iloc[i, j] < alpha:
                sig_pairs.append({'Var1': cols[i], 'Var2': cols[j],
                                   'Corr': corr_matrix.iloc[i, j],
                                   'P':    p_matrix.iloc[i, j]})
    sig_df = pd.DataFrame(sig_pairs)
    strong = (sig_df[abs(sig_df['Corr']) > threshold]
              if not sig_df.empty else pd.DataFrame())

    condensed = squareform(1 - abs(corr_matrix))
    Z = linkage(condensed, method='average')

    fig1, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title(f'Correlation Matrix ({corr_method})')

    fig2, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=cols, leaf_rotation=90, ax=ax)
    ax.set_title('Clustered Dendrogram')

    return {'correlation_matrix': corr_matrix,
            'p_value_matrix':    p_matrix,
            'significant_pairs': sig_df,
            'strong_correlations': strong,
            'figures': [('corr_heatmap.png', fig1), ('dendrogram.png', fig2)]}

# 2. REGRESSION
def regression_analysis(data, y_var, x_vars, reg_type='multiple',
                        test_size=0.2, poly_deg=2, ridge_alpha=1.0):
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score

    X = data[x_vars].dropna()
    y = data.loc[X.index, y_var]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    if reg_type == 'simple' and len(x_vars) == 1:
        X_train_sm = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_sm).fit()
        y_pred_test  = model.predict(sm.add_constant(X_test))
        y_pred_train = model.predict(X_train_sm)
        coef_df = pd.DataFrame({'Variable': ['Intercept'] + x_vars,
                                'Coef':     model.params.values,
                                'p-value':  model.pvalues.values})
        r2 = model.rsquared; r2_adj = model.rsquared_adj
        eq = (f"{y_var} = {model.params.values[0]:.3f} + "
              f"{model.params.values[1]:.3f}*{x_vars[0]}")

    elif reg_type == 'polynomial':
        poly = PolynomialFeatures(degree=poly_deg, include_bias=False)
        X_tr_p = poly.fit_transform(X_train)
        X_te_p = poly.transform(X_test)
        model = sm.OLS(y_train, sm.add_constant(X_tr_p)).fit()
        y_pred_test  = model.predict(sm.add_constant(X_te_p))
        y_pred_train = model.predict(sm.add_constant(X_tr_p))
        coef_df = pd.DataFrame({
            'Variable': ['Intercept'] + [f'poly_{i}' for i in range(X_tr_p.shape[1])],
            'Coef': model.params.values})
        r2 = model.rsquared; r2_adj = model.rsquared_adj
        eq = f"Polynomial degree {poly_deg}"

    elif reg_type == 'ridge':
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_tr_s, y_train)
        y_pred_test  = ridge.predict(X_te_s)
        y_pred_train = ridge.predict(X_tr_s)
        coef_df = pd.DataFrame({'Variable': x_vars, 'Coef': ridge.coef_})
        r2     = ridge.score(X_tr_s, y_train)
        r2_adj = 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - len(x_vars) - 1)
        eq = (f"{y_var} = {ridge.intercept_:.3f} + " +
              " + ".join([f"{c:.3f}*{v}" for v, c in zip(x_vars, ridge.coef_)]))

    elif reg_type == 'lasso':
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        lasso = Lasso(alpha=ridge_alpha)
        lasso.fit(X_tr_s, y_train)
        y_pred_test  = lasso.predict(X_te_s)
        y_pred_train = lasso.predict(X_tr_s)
        coef_df = pd.DataFrame({'Variable': x_vars, 'Coef': lasso.coef_})
        r2     = lasso.score(X_tr_s, y_train)
        r2_adj = 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - len(x_vars) - 1)
        eq = (f"{y_var} = {lasso.intercept_:.3f} + " +
              " + ".join([f"{c:.3f}*{v}" for v, c in zip(x_vars, lasso.coef_)]))

    else:  # multiple linear (default)
        X_train_sm = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_sm).fit()
        y_pred_test  = model.predict(sm.add_constant(X_test))
        y_pred_train = model.predict(X_train_sm)
        coef_df = pd.DataFrame({'Variable': ['Intercept'] + x_vars,
                                'Coef':    model.params.values,
                                'StdErr':  model.bse.values,
                                't':       model.tvalues.values,
                                'p-value': model.pvalues.values})
        r2 = model.rsquared; r2_adj = model.rsquared_adj
        eq = (f"{y_var} = {model.params.values[0]:.3f} + " +
              " + ".join([f"{c:.3f}*{v}"
                          for v, c in zip(x_vars, model.params.values[1:])]))

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    r2_test    = r2_score(y_test, y_pred_test)
    perf = pd.DataFrame({'Dataset': ['Train', 'Test'],
                         'RMSE':    [rmse_train, rmse_test],
                         'R²':      [r2, r2_test]})

    residuals = y_test - y_pred_test
    fig1, ax = plt.subplots()
    ax.scatter(y_pred_test, residuals); ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')

    fig2, ax = plt.subplots()
    sm.qqplot(residuals, line='s', ax=ax); ax.set_title('Q-Q Plot')

    fig3, ax = plt.subplots()
    ax.hist(residuals, bins=20, edgecolor='black'); ax.set_title('Residual Histogram')

    fig4, ax = plt.subplots()
    ax.scatter(y_test, y_pred_test, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')

    if len(x_vars) >= 2:
        vif_df = pd.DataFrame({
            'Variable': x_vars,
            'VIF': [variance_inflation_factor(X.values, i) for i in range(len(x_vars))]})
    else:
        vif_df = pd.DataFrame({'Variable': x_vars,
                               'VIF': ['N/A — single predictor']})

    return {'model_summary': pd.DataFrame({'Metric': ['R²', 'Adj R²'],
                                           'Value':  [r2, r2_adj]}),
            'coefficients': coef_df,
            'vif':          vif_df,
            'performance':  perf,
            'equation':     eq,
            'figures': [('resid_fitted.png', fig1), ('qq.png', fig2),
                        ('resid_hist.png',   fig3), ('actual_vs_pred.png', fig4)]}

# 3. HIERARCHICAL CLUSTERING
def hierarchical_clustering(data, vars_list, linkage_method='ward',
                            dist_metric='euclidean', n_clusters=3):
    X = data[vars_list].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dist = pdist(X_scaled, metric=dist_metric)
    Z = linkage(dist, method=linkage_method)
    coph, _ = cophenet(Z, dist)
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    sil = silhouette_score(X_scaled, labels, metric=dist_metric)
    membership = pd.DataFrame({'Sample': X.index, 'Cluster': labels})
    summary = X.copy(); summary['Cluster'] = labels
    summary = summary.groupby('Cluster').agg(['mean', 'std', 'count']).round(4)

    fig1, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, labels=X.index.tolist(), leaf_rotation=90)
    ax.set_title(f'Dendrogram ({linkage_method})')

    pca = PCA(2)
    pca_res = pca.fit_transform(X_scaled)
    fig2, ax = plt.subplots()
    scatter = ax.scatter(pca_res[:, 0], pca_res[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter); ax.set_title('Cluster Scatter (PCA)')

    return {'membership':      membership,
            'cluster_summary': summary,
            'cophenetic_corr': coph,
            'silhouette_score': sil,
            'figures': [('dendrogram.png', fig1), ('cluster_scatter.png', fig2)]}

# 4. K-MEANS CLUSTERING
def kmeans_clustering(data, vars_list, n_clusters=3, random_seed=42):
    X = data[vars_list].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    membership = pd.DataFrame({'Sample': X.index, 'Cluster': labels})
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=vars_list)
    summary = X.copy(); summary['Cluster'] = labels
    summary = summary.groupby('Cluster').agg(['mean', 'std', 'count']).round(4)

    inertias = []
    Ks = range(2, min(11, X_scaled.shape[0] - 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        km.fit(X_scaled); inertias.append(km.inertia_)
    fig1, ax = plt.subplots()
    ax.plot(Ks, inertias, 'bo-')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title('Elbow Plot')

    fig2, ax = plt.subplots(figsize=(10, 6))
    sil_vals = silhouette_samples(X_scaled, labels); y_lower = 10
    for i in range(n_clusters):
        c_sil = sil_vals[labels == i]; c_sil.sort()
        y_upper = y_lower + len(c_sil)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil)
        ax.text(-0.05, y_lower + 0.5 * len(c_sil), str(i))
        y_lower = y_upper + 10
    ax.axvline(sil, color='red', linestyle='--', label=f'Avg: {sil:.3f}')
    ax.set_xlabel('Silhouette'); ax.set_ylabel('Cluster'); ax.legend()

    pca = PCA(2); pca_res = pca.fit_transform(X_scaled)
    fig3, ax = plt.subplots()
    scatter = ax.scatter(pca_res[:, 0], pca_res[:, 1], c=labels, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200)
    plt.colorbar(scatter); ax.set_title('K-Means Clusters')

    return {'membership':      membership,
            'cluster_summary': summary,
            'centers':         centers,
            'inertia':         kmeans.inertia_,
            'silhouette_score': sil,
            'figures': [('elbow.png', fig1), ('silhouette.png', fig2), ('scatter.png', fig3)]}

# 5. LDA / QDA
def discriminant_analysis(data, x_vars, y_var, da_type='lda', test_size=0.2):
    X = data[x_vars].dropna()
    y_raw = data.loc[X.index, y_var]
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = (LinearDiscriminantAnalysis() if da_type == 'lda'
             else QuadraticDiscriminantAnalysis())
    model.fit(X_tr_s, y_train)
    y_pred = model.predict(X_te_s)
    acc = model.score(X_te_s, y_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    report = classification_report(y_test, y_pred,
                                   target_names=[str(c) for c in le.classes_],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    centroids = pd.DataFrame(model.means_, columns=x_vars, index=le.classes_)

    fig1, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')

    X_all_s = scaler.transform(X)
    if da_type == 'lda' and hasattr(model, 'coef_') and model.coef_.shape[0] >= 2:
        X_lda = model.transform(X_all_s)
        fig2, ax = plt.subplots()
        for i, cls in enumerate(le.classes_):
            mask = (y == i)
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], label=cls, alpha=0.6)
        ax.set_xlabel('LD1'); ax.set_ylabel('LD2'); ax.legend()
    else:
        pca = PCA(2); X_pca = pca.fit_transform(X_all_s)
        fig2, ax = plt.subplots()
        for i, cls in enumerate(le.classes_):
            mask = (y == i)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls, alpha=0.6)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend()
    ax.set_title('Discriminant / PCA Projection')

    if da_type == 'lda':
        coef_df = pd.DataFrame({'Variable':    x_vars,
                                'Coefficient': (model.coef_[0]
                                                if model.coef_.ndim > 1
                                                else model.coef_)})
        fig3, ax = plt.subplots()
        coef_df.sort_values('Coefficient').plot.barh(
            x='Variable', y='Coefficient', ax=ax)
        ax.set_title('Standardized Coefficients')
        return {'confusion_matrix':       cm_df,
                'classification_report':  report_df,
                'coefficients':           coef_df,
                'group_centroids':        centroids,
                'accuracy':               acc,
                'figures': [('confusion_matrix.png', fig1),
                            ('score_plot.png',       fig2),
                            ('coef_bar.png',         fig3)]}
    return {'confusion_matrix':      cm_df,
            'classification_report': report_df,
            'group_centroids':       centroids,
            'accuracy':              acc,
            'figures': [('confusion_matrix.png', fig1), ('score_plot.png', fig2)]}

# 6. FACTOR ANALYSIS (EFA)
def factor_analysis(data, vars_list, n_factors='auto', rotation='varimax'):
    X = data[vars_list].dropna()
    n, p = X.shape

    # KMO
    R = np.corrcoef(X.T)
    R_inv = np.linalg.pinv(R)
    partial = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            partial[i, j] = (-R_inv[i, j] /
                             np.sqrt(abs(R_inv[i, i] * R_inv[j, j])))
    np.fill_diagonal(partial, 0)
    R_off = R.copy(); np.fill_diagonal(R_off, 0)
    kmo_model = np.sum(R_off ** 2) / (np.sum(R_off ** 2) + np.sum(partial ** 2))

    # Bartlett's test
    chi2_b = -(n - 1 - (2 * p + 5) / 6) * np.log(max(np.linalg.det(R), 1e-15))
    df_b = p * (p - 1) // 2
    bartlett_p = 1 - stats.chi2.cdf(chi2_b, df_b)

    pca_obj = PCA().fit(StandardScaler().fit_transform(X))
    ev = pca_obj.explained_variance_

    if n_factors == 'auto':
        n_factors = max(int(sum(ev > 1)), 2)

    fa = SKLearnFA(n_components=n_factors, random_state=42)
    fa.fit(X)
    loadings = pd.DataFrame(fa.components_.T,
                            index=vars_list,
                            columns=[f'F{i + 1}' for i in range(n_factors)])
    communalities = np.sum(fa.components_ ** 2, axis=0)
    comm_df = pd.DataFrame({'Variable': vars_list, 'Communality': communalities})

    var_exp_pct = ev[:n_factors] / ev.sum() * 100
    var_df = pd.DataFrame({
        'Factor':     range(1, n_factors + 1),
        'Eigenvalue': ev[:n_factors],
        'Var%':       var_exp_pct,
        'CumVar%':    np.cumsum(var_exp_pct)})

    fig1, ax = plt.subplots()
    ax.scatter(range(1, len(ev) + 1), ev)
    ax.plot(range(1, len(ev) + 1), ev)
    ax.axhline(1, color='r', linestyle='--')
    ax.set_xlabel('Factor'); ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree Plot')

    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Factor Loadings')

    return {'eigenvalues':        pd.DataFrame({'Factor': range(1, len(ev) + 1),
                                                'Eigenvalue': ev}),
            'loadings':           loadings,
            'variance_explained': var_df,
            'communalities':      comm_df,
            'kmo':                kmo_model,
            'bartlett_p':         bartlett_p,
            'figures': [('scree.png', fig1), ('loadings_heatmap.png', fig2)]}

# 7. MANOVA
def manova_analysis(data, dep_vars, group_var):
    data_clean = data[dep_vars + [group_var]].dropna()
    formula = " + ".join(dep_vars) + f" ~ C({group_var})"
    manova = MANOVA.from_formula(formula, data=data_clean)
    result = manova.mv_test()

    manova_key = f"C({group_var})"
    stat_table = result[manova_key]['stat']
    wilks = stat_table.iloc[0]
    wilks_df = pd.DataFrame({
        'Statistic': ['Wilks λ', 'F', 'Num DF', 'Den DF', 'Pr>F'],
        'Value':     [wilks['Value'], wilks['F Value'],
                      wilks['Num DF'], wilks['Den DF'], wilks['Pr > F']]})

    anova_list = []
    for var in dep_vars:
        m = ols(f"{var} ~ C({group_var})", data=data_clean).fit()
        aov = sm.stats.anova_lm(m, typ=2)
        eta = aov['sum_sq'].iloc[0] / (aov['sum_sq'].iloc[0] + aov['sum_sq'].iloc[1])
        anova_list.append({'Variable': var,
                           'F':        aov['F'].iloc[0],
                           'p':        aov['PR(>F)'].iloc[0],
                           'η²':       eta})
    anova_df = pd.DataFrame(anova_list)
    group_means = data_clean.groupby(group_var)[dep_vars].mean()

    figs = []
    for var in dep_vars:
        fig, ax = plt.subplots()
        data_clean.boxplot(column=var, by=group_var, ax=ax)
        ax.set_title(f'Boxplot of {var}')
        figs.append((f'boxplot_{var}.png', fig))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(group_means, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Group Means Heatmap')
    figs.append(('means_heatmap.png', fig))

    return {'manova_summary':   wilks_df,
            'univariate_anova': anova_df,
            'group_means':      group_means,
            'figures':          figs}

# 8. LOGISTIC REGRESSION
def logistic_regression(data, y_var, x_vars, test_size=0.2):
    X = data[x_vars].dropna()
    y_raw = data.loc[X.index, y_var]
    unique = y_raw.unique()
    if len(unique) != 2:
        raise ValueError("Binary outcome required.")
    y = (y_raw == unique[1]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_tr_sm = sm.add_constant(X_train)
    X_te_sm = sm.add_constant(X_test)
    model = sm.Logit(y_train, X_tr_sm).fit(disp=0)
    y_pred_prob = model.predict(X_te_sm)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=['Pred 0', 'Pred 1'],
                         index=['Actual 0', 'Actual 1'])
    coef = pd.DataFrame({'Variable': ['Intercept'] + x_vars,
                         'Coef':     model.params.values,
                         'OR':       np.exp(model.params.values),
                         'p-value':  model.pvalues.values})
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig1, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend()

    fig2, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')

    return {'coefficients':    coef,
            'confusion_matrix': cm_df,
            'accuracy':        acc,
            'roc_auc':         roc_auc,
            'figures': [('roc.png', fig1), ('confusion.png', fig2)]}

# 9. CANONICAL CORRELATION
def canonical_correlation(data, set1_vars, set2_vars, n_components='auto'):
    X = data[set1_vars].dropna()
    Y = data[set2_vars].loc[X.index]
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    n_comp = min(len(set1_vars), len(set2_vars), Xs.shape[0] - 1)
    if n_components != 'auto':
        n_comp = min(n_components, n_comp)
    cca = CCA(n_components=n_comp)
    cca.fit(Xs, Ys)
    Xc, Yc = cca.transform(Xs, Ys)
    corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n_comp)]

    n = Xs.shape[0]; p = len(set1_vars); q = len(set2_vars); res = []
    for k in range(n_comp):
        wilks = np.prod([1 - corrs[i] ** 2 for i in range(k, n_comp)])
        df = (p - k) * (q - k)
        chi2_v = -(n - 0.5 * (p + q + 1)) * np.log(max(wilks, 1e-15))
        pval = 1 - stats.chi2.cdf(chi2_v, df)
        res.append({'Canonical Variate': k + 1,
                    'Rc':      corrs[k],
                    'Rc²':     corrs[k] ** 2,
                    'Wilks λ': wilks,
                    'Chi2':    chi2_v,
                    'df':      df,
                    'p':       pval})
    tab = pd.DataFrame(res)

    load_x = np.corrcoef(Xs.T, Xc.T)[:Xs.shape[1], Xs.shape[1]:]
    load_y = np.corrcoef(Ys.T, Yc.T)[:Ys.shape[1], Ys.shape[1]:]
    load_x_df = pd.DataFrame(load_x, index=set1_vars,
                              columns=[f'CV{i + 1}' for i in range(n_comp)])
    load_y_df = pd.DataFrame(load_y, index=set2_vars,
                              columns=[f'CV{i + 1}' for i in range(n_comp)])

    fig, ax = plt.subplots()
    ax.scatter(Xc[:, 0], Yc[:, 0], alpha=0.6)
    ax.set_xlabel('CV1 (X)'); ax.set_ylabel('CV1 (Y)')
    ax.set_title(f'CCA: r={corrs[0]:.3f}')

    return {'canonical_correlations': tab,
            'x_loadings':            load_x_df,
            'y_loadings':            load_y_df,
            'figures': [('cca_scatter.png', fig)]}

# 10. t-SNE / UMAP
def tsne_umap(data, vars_list, color_var=None, method='tsne',
              perplexity=30, n_neighbors=15):
    X = data[vars_list].dropna()
    Xs = StandardScaler().fit_transform(X)
    if color_var and color_var in data.columns:
        colors = data.loc[X.index, color_var]
        if colors.dtype == 'object':
            colors = LabelEncoder().fit_transform(colors.astype(str))
    else:
        colors = np.zeros(len(Xs))

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       random_state=42, max_iter=1000)
        emb = reducer.fit_transform(Xs)
        title = f't-SNE (perp={perplexity})'
    else:
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        emb = reducer.fit_transform(Xs)
        title = f'UMAP (n_neigh={n_neighbors})'

    sil = (silhouette_score(emb, colors)
           if len(np.unique(colors)) > 1 else np.nan)
    coord = pd.DataFrame(emb, columns=['Dim1', 'Dim2'], index=X.index)

    fig, ax = plt.subplots()
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap='viridis', alpha=0.7)
    if color_var:
        plt.colorbar(sc, label=color_var)
    ax.set_title(title)

    return {'coordinates':     coord,
            'silhouette_score': sil,
            'figures': [('dimred_plot.png', fig)]}

# 11. SEM
def sem_analysis(data, model_spec, observed_vars):
    data_sub = data[observed_vars].dropna()
    model = semopy.Model(model_spec)
    try:
        fit = model.fit(data_sub, solver='SLSQP')
        estimates = model.inspect(fit)
        from semopy import calc_stats
        stats_out = calc_stats(model)
        stats_df = pd.DataFrame(stats_out.items(), columns=['Index', 'Value'])
        params = estimates[estimates['op'] != '~~']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                "Path diagram not auto-generated.\nSee parameter table.",
                ha='center', va='center')
        ax.axis('off')
        return {'parameter_estimates': params,
                'fit_indices':         stats_df,
                'converged':           True,
                'figures': [('sem_info.png', fig)]}
    except Exception as e:
        return {'error': str(e), 'converged': False}

# --------------------------------------------------------------------
#  STREAMLIT APP INTERFACE
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multivariate Analysis Tool", layout="wide")
    st.title("📊 Multivariate Analysis Tool")
    st.markdown("Upload your dataset (CSV or Excel) and choose an analysis method.")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to begin.")
        return

    # Read data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.success(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    st.write("**Preview of your data:**")
    st.dataframe(data.head())

    # Sidebar: method selection
    st.sidebar.header("Analysis Method")
    method_choice = st.sidebar.selectbox(
        "Select a method",
        [
            "Correlation Analysis",
            "Multiple Regression",
            "Hierarchical Clustering",
            "K-Means Clustering",
            "LDA / QDA",
            "Factor Analysis (EFA)",
            "MANOVA",
            "Logistic Regression",
            "Canonical Correlation (CCA)",
            "t-SNE / UMAP",
            "SEM (Structural Equation Modeling)"
        ]
    )

    # Container for dynamic inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameters")

    # We'll collect parameters in a dictionary and then call the appropriate function
    params = {}
    run_analysis = st.sidebar.button("▶ Run Analysis", type="primary")

    # Build parameter inputs based on method (shown only when method selected)
    if method_choice == "Correlation Analysis":
        numeric_cols = get_numeric_cols(data)
        vars_list = st.sidebar.multiselect("Variables", numeric_cols)
        corr_method = st.sidebar.selectbox("Correlation type", ["pearson", "spearman", "kendall"])
        alpha = st.sidebar.number_input("Significance level (alpha)", 0.01, 0.20, 0.05, 0.01)
        threshold = st.sidebar.number_input("Threshold for strong correlations", 0.0, 1.0, 0.5, 0.05)
        params = {"vars_list": vars_list, "corr_method": corr_method,
                  "alpha": alpha, "threshold": threshold}
        func = correlation_analysis

    elif method_choice == "Multiple Regression":
        numeric_cols = get_numeric_cols(data)
        y_var = st.sidebar.selectbox("Dependent variable (Y)", numeric_cols)
        x_vars = st.sidebar.multiselect("Independent variables (X)", numeric_cols, default=[c for c in numeric_cols if c != y_var][:2])
        reg_type = st.sidebar.selectbox("Regression type", ["multiple", "simple", "polynomial", "ridge", "lasso"])
        test_size = st.sidebar.slider("Test proportion", 0.1, 0.4, 0.2, 0.05)
        poly_deg = 2
        ridge_alpha = 1.0
        if reg_type == "polynomial":
            poly_deg = st.sidebar.number_input("Polynomial degree", 2, 5, 2)
        if reg_type in ["ridge", "lasso"]:
            ridge_alpha = st.sidebar.number_input("Alpha (regularisation strength)", 0.01, 10.0, 1.0)
        params = {"y_var": y_var, "x_vars": x_vars, "reg_type": reg_type,
                  "test_size": test_size, "poly_deg": poly_deg, "ridge_alpha": ridge_alpha}
        func = regression_analysis

    elif method_choice == "Hierarchical Clustering":
        numeric_cols = get_numeric_cols(data)
        vars_list = st.sidebar.multiselect("Variables", numeric_cols)
        linkage_meth = st.sidebar.selectbox("Linkage method", ["ward", "complete", "average", "single"])
        dist_met = st.sidebar.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"])
        n_clust = st.sidebar.number_input("Number of clusters", 2, 10, 3)
        params = {"vars_list": vars_list, "linkage_method": linkage_meth,
                  "dist_metric": dist_met, "n_clusters": n_clust}
        func = hierarchical_clustering

    elif method_choice == "K-Means Clustering":
        numeric_cols = get_numeric_cols(data)
        vars_list = st.sidebar.multiselect("Variables", numeric_cols)
        n_clust = st.sidebar.number_input("Number of clusters", 2, 10, 3)
        seed = st.sidebar.number_input("Random seed", 0, 999, 42)
        params = {"vars_list": vars_list, "n_clusters": n_clust, "random_seed": seed}
        func = kmeans_clustering

    elif method_choice == "LDA / QDA":
        numeric_cols = get_numeric_cols(data)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        y_var = st.sidebar.selectbox("Group variable (categorical)", categorical_cols + numeric_cols)
        x_vars = st.sidebar.multiselect("Predictor variables", numeric_cols)
        da_type = st.sidebar.selectbox("DA type", ["lda", "qda"])
        test_size = st.sidebar.slider("Test proportion", 0.1, 0.4, 0.2, 0.05)
        params = {"x_vars": x_vars, "y_var": y_var, "da_type": da_type, "test_size": test_size}
        func = discriminant_analysis

    elif method_choice == "Factor Analysis (EFA)":
        numeric_cols = get_numeric_cols(data)
        vars_list = st.sidebar.multiselect("Variables", numeric_cols)
        n_fact = st.sidebar.text_input("Number of factors (auto or integer)", "auto")
        if n_fact.isdigit():
            n_fact = int(n_fact)
        rotation = st.sidebar.selectbox("Rotation", ["varimax", "promax", "None"])
        if rotation == "None":
            rotation = None
        params = {"vars_list": vars_list, "n_factors": n_fact, "rotation": rotation}
        func = factor_analysis

    elif method_choice == "MANOVA":
        numeric_cols = get_numeric_cols(data)
        dep_vars = st.sidebar.multiselect("Dependent variables", numeric_cols)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        group_var = st.sidebar.selectbox("Grouping variable", categorical_cols + numeric_cols)
        params = {"dep_vars": dep_vars, "group_var": group_var}
        func = manova_analysis

    elif method_choice == "Logistic Regression":
        # Binary outcome: need to check if selected variable has two unique values
        all_cols = data.columns.tolist()
        y_var = st.sidebar.selectbox("Binary outcome variable", all_cols)
        x_vars = st.sidebar.multiselect("Predictor variables", get_numeric_cols(data))
        test_size = st.sidebar.slider("Test proportion", 0.1, 0.4, 0.2, 0.05)
        params = {"y_var": y_var, "x_vars": x_vars, "test_size": test_size}
        func = logistic_regression

    elif method_choice == "Canonical Correlation (CCA)":
        numeric_cols = get_numeric_cols(data)
        set1 = st.sidebar.multiselect("Set X variables", numeric_cols)
        set2 = st.sidebar.multiselect("Set Y variables", numeric_cols)
        n_comp = st.sidebar.text_input("Number of components (auto or integer)", "auto")
        if n_comp.isdigit():
            n_comp = int(n_comp)
        params = {"set1_vars": set1, "set2_vars": set2, "n_components": n_comp}
        func = canonical_correlation

    elif method_choice == "t-SNE / UMAP":
        numeric_cols = get_numeric_cols(data)
        vars_list = st.sidebar.multiselect("Variables", numeric_cols)
        all_cols = data.columns.tolist()
        color_var = st.sidebar.selectbox("Color by variable (optional)", ["None"] + all_cols)
        if color_var == "None":
            color_var = None
        method_red = st.sidebar.selectbox("Method", ["tsne", "umap"])
        perplexity = 30
        n_neighbors = 15
        if method_red == "tsne":
            perplexity = st.sidebar.number_input("Perplexity", 5, 100, 30)
        else:
            n_neighbors = st.sidebar.number_input("n_neighbors", 5, 50, 15)
        params = {"vars_list": vars_list, "color_var": color_var, "method": method_red,
                  "perplexity": perplexity, "n_neighbors": n_neighbors}
        func = tsne_umap

    elif method_choice == "SEM (Structural Equation Modeling)":
        observed_vars = st.sidebar.multiselect("Observed variables", data.columns.tolist())
        model_spec = st.sidebar.text_area("Model specification (e.g., 'y1 ~ x1 + x2')")
        params = {"model_spec": model_spec, "observed_vars": observed_vars}
        func = sem_analysis

    # Run analysis when button clicked
    if run_analysis:
        if method_choice in ["Correlation Analysis", "Hierarchical Clustering", "K-Means Clustering", "Factor Analysis (EFA)"]:
            if not params.get("vars_list"):
                st.error("Please select at least one variable.")
                return
        if method_choice == "Multiple Regression":
            if not params.get("x_vars") or not params.get("y_var"):
                st.error("Please select both dependent and independent variables.")
                return
        if method_choice in ["LDA / QDA", "Logistic Regression"]:
            if not params.get("x_vars") or not params.get("y_var"):
                st.error("Please select predictors and outcome variable.")
                return
        if method_choice == "MANOVA":
            if not params.get("dep_vars") or not params.get("group_var"):
                st.error("Please select dependent variables and a grouping variable.")
                return
        if method_choice == "Canonical Correlation (CCA)":
            if not params.get("set1_vars") or not params.get("set2_vars"):
                st.error("Please select variables for both sets.")
                return
        if method_choice == "SEM (Structural Equation Modeling)":
            if not params.get("observed_vars") or not params.get("model_spec"):
                st.error("Please provide observed variables and a model specification.")
                return

        with st.spinner("Running analysis..."):
            try:
                results = func(data, **params)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return

        # Display results
        st.success("Analysis completed!")
        st.subheader("Results")

        # Show key tables
        for key, val in results.items():
            if key == 'figures':
                continue
            if isinstance(val, pd.DataFrame):
                st.write(f"**{key.replace('_', ' ').title()}**")
                st.dataframe(val)
            elif isinstance(val, (int, float, str, np.number)):
                st.metric(key.replace('_', ' ').title(), val)

        # Show plots
        if 'figures' in results and results['figures']:
            st.subheader("Plots")
            for name, fig in results['figures']:
                st.pyplot(fig)
                plt.close(fig)

        # Download section: Excel + ZIP
        st.subheader("Download Results")

        # Create Excel file
        output_xlsx = io.BytesIO()
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            for key, val in results.items():
                if key == 'figures':
                    continue
                if isinstance(val, pd.DataFrame):
                    val.to_excel(writer, sheet_name=key[:31])
                elif isinstance(val, (int, float, str, np.number)):
                    pd.DataFrame({'Value': [val]}).to_excel(writer, sheet_name=key[:31])
                elif isinstance(val, dict):
                    pd.DataFrame(val).to_excel(writer, sheet_name=key[:31])
        output_xlsx.seek(0)

        st.download_button(
            label="📊 Download Excel Results",
            data=output_xlsx,
            file_name=f"{method_choice.replace(' ', '_')}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Create ZIP of plots
        if 'figures' in results and results['figures']:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for name, fig in results['figures']:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    zf.writestr(name, buf.getvalue())
            zip_buf.seek(0)
            st.download_button(
                label="📈 Download All Plots (ZIP)",
                data=zip_buf,
                file_name=f"{method_choice.replace(' ', '_')}_plots.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
