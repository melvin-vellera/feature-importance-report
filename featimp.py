from cmath import inf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import joblib
import xgboost
import shap
import pandas as pd
from sklearn.base import clone
from scipy.stats import f_oneway, rankdata, spearmanr
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

def spearman_rank_correlation(x, y, method='average'):
    # Convert to ranks and compute pearson correlation
    if method =='ordinal':
        x_r = np.argsort(x) + 1 #method = ordinal
        y_r = np.argsort(y) + 1
    else:
        x_r = rankdata(x)       #method = average
        y_r = rankdata(y)
    return pearson_correlation(x_r, y_r)

def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    s_xy = np.sum((x - x_mean)*(y - y_mean))
    s_xx = np.sum((x - x_mean)**2)
    s_yy = np.sum((y - y_mean)**2)
    return s_xy / np.sqrt(s_xx*s_yy)    

def read_dataset(fetch_dataset):
    data = fetch_dataset()
    df = pd.concat([
        pd.DataFrame(data.data, columns=data.feature_names),
        pd.DataFrame(data.target, columns=[data.target_names[0]])
    ], axis=1)
    X = df.drop([data.target_names[0]], axis=1)
    y = df[[data.target_names[0]]]    
    return X, y

def get_loadings(pca, X):
    return pd.DataFrame(
        data=np.abs(pca.components_.T) * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X.columns) + 1)],
        index=X.columns
    )

def hbar(x, y, xlabel, ylabel, title):
    plt.barh(x, y)
    # plt.xticks(rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

class MRMR:
    def __init__(self, df, target_name):
        self.df = df
        self.class_idxs = [df[df[target_name] == v].index for v in df[target_name].unique()]
        self.features = [col for col in df.columns if col != target_name]
        self.selected_features = {}
        self.feature_relevance = {feat_name: self.feature_relevance(self.df[feat_name]) 
                                  for feat_name in
                                  self.features}
        self.pair_correlations = {}

    def feature_relevance(self, feature):
        groups = [feature[class_idxs].values for class_idxs in self.class_idxs]
        return f_oneway(*groups).statistic

    def feature_redundancy(self, feature):
        redundancy = 0
        for feat in self.selected_features:
            if (feat, feature) not in self.pair_correlations:        
                self.pair_correlations[(feat, feature)] = abs(spearman_rank_correlation(self.df[feature], self.df[feat]))
                self.pair_correlations[(feature, feat)] = self.pair_correlations[(feat, feature)]
            redundancy += self.pair_correlations[(feat, feature)]
        return redundancy / len(self.selected_features)

    def feature_importance(self):
        most_important_feature, max_importance = max(self.feature_relevance.items(), key=lambda x: x[1])
        self.selected_features[most_important_feature] = max_importance

        while len(self.selected_features) != len(self.features):
            max_importance = float('-inf')
            most_important_feature = None
            for feat in self.features:
                if feat in self.selected_features:
                    continue

                feature_redundancy = self.feature_redundancy(feat)
                feature_relevance = self.feature_relevance[feat]
                
                importance = feature_relevance / feature_redundancy
                
                if importance > max_importance:
                    max_importance = importance
                    most_important_feature = feat

            self.selected_features[most_important_feature] = max_importance

        return self.selected_features

def permutation_importances(model, X_valid, y_valid, metric=accuracy_score): 
    baseline = metric(y_valid, model.predict(X_valid.values)) 
    importances = []
    for col in X_valid.columns:
        temp = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        current = metric(y_valid, model.predict(X_valid.values)) 
        X_valid[col] = temp
        importances.append(baseline - current)
    return importances

def dropcol_importances(model, X_train, y_train, X_valid, y_valid, metric=accuracy_score):
    model.fit(X_train.values, y_train)
    baseline = metric(y_valid, model.predict(X_valid.values)) 
    importances = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1) 
        X_valid_ = X_valid.drop(col, axis=1) 
        model_ = clone(model) 
        model_.fit(X_train_.values, y_train)
        current = metric(y_valid, model_.predict(X_valid_.values))
        importances.append(baseline - current) 
    return importances            


def automatic_feature_selection(model, X_train, y_train, X_valid, y_valid, metric, algorithm='permutation'):
    feature_names = list(X_train.columns)
    print(f"Starting with {len(feature_names)} features: {feature_names}\n")
    model.fit(X_train.values, y_train)
    metric_value = metric(y_valid, model.predict(X_valid.values)) 
    while len(feature_names):
        print(f"Validation Score for {len(feature_names)} features: {metric_value:.4f}")
        if algorithm == 'permutation':
            imp = pd.DataFrame({'feature': X_train.columns.values, 
                                'importance': permutation_importances(model, X_valid, y_valid, metric)})
        elif algorithm == 'drop':
            imp = pd.DataFrame({'feature': X_train.columns.values, 
                                'importance': dropcol_importances(model, X_train, y_train, X_valid, y_valid, metric)})
        imp = imp.sort_values('importance', ascending=False) 
        feature_to_remove = imp.iloc[-1]['feature']   
        X_train_ = X_train.drop(feature_to_remove, axis=1)
        X_valid_ = X_valid.drop(feature_to_remove, axis=1)  
        model_ = clone(model) 
        model_.fit(X_train_.values, y_train)
        new_metric_value = metric(y_valid, model_.predict(X_valid_.values))      
        if new_metric_value > metric_value: # model improvement after feature removal
            X_train = X_train_
            X_valid = X_valid_
            metric_value = new_metric_value
            model = model_
            feature_names.remove(feature_to_remove)
            print(f'Removed feature {feature_to_remove}')
        else:
            break
    print(f"\nFeatures selected: {feature_names}")
    return feature_names

def pca_importance(X_train):
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X_train)

    pca = PCA().fit(X_scaled)
    loadings = get_loadings(pca, X_train)

    # loadings_sum = loadings.sum(axis=1).sort_values(ascending=False)
    # hbar(loadings_sum.index, loadings_sum, 'Importance', 'Features', 'Feature Importance (PCA)')

    pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['feature', 'importance']
    return pc1_loadings

def get_train_test_split(df, feature_names, target_name):
    df_train, df_test = train_test_split(df[feature_names + [target_name]], test_size=0.2, random_state=1)
    X_train, y_train = df_train.drop(target_name,axis=1), df_train[target_name]
    X_test, y_test = df_test.drop(target_name,axis=1), df_test[target_name]
    return X_train, y_train, X_test, y_test

def feature_df(features, importances):
    importances_df = pd.DataFrame({'feature': features, 
                                'importance': importances})
    importances_df = importances_df.sort_values('importance', ascending=False)
    importances_df = importances_df.reset_index(drop=True) 
    return importances_df

def spearman_importance(X, y):
    imps = []
    features = X.columns.values
    for col in features:
        imp = spearman_rank_correlation(X[col], y)
        imps.append(abs(imp))
    spearman_imp = feature_df(features, imps)     
    return spearman_imp

def process_rent_json(path="train.json"):
    df = pd.read_json(path)
    df = df[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'interest_level']].copy()
    encoding = {'high': 3, 'medium': 2, 'low': 1}
    df.loc[:, 'interest_level']  = [encoding[w] for w in df['interest_level']]
    # df.to_csv('rent.csv', index=False)
    return df

def drop_column(train, test, column):
    train = train.drop(column, axis=1)
    test = test.drop(column, axis=1)
    return train, test

def reset_df(X_train_c, X_test_c, X_train_r, X_test_r, classification_features, regression_features):
    X_train_c = X_train_c[classification_features]
    X_test_c = X_test_c[classification_features]
    X_train_r = X_train_r[regression_features]
    X_test_r = X_test_r[regression_features]
    return X_train_c, X_test_c, X_train_r, X_test_r

def feature_df_shap(feature_names, shap_values):
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)
    importances_df = pd.DataFrame(list(zip(feature_names, vals)), 
                                    columns=['feature', 'importance'])
    
    importances_df.sort_values(by=['importance'], ascending=False, inplace=True)                                    
    importances_df = importances_df.reset_index(drop=True) 
    return importances_df

def hbar_ax(x, y, xlabel, ylabel, title, ax):
    ax.barh(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()


def shap_XGB_RF(X_train_r, y_train_r):
    # Fit XGBoost model
    xgb = xgboost.XGBRegressor(min_child_weight=5, n_jobs=-1).fit(X_train_r, y_train_r)

    # Get XGB shap values
    xgb_explainer = shap.Explainer(xgb)
    xgb_shap_values = xgb_explainer(X_train_r)

    # # Fit Random Forest Regressor
    # rf = RandomForestRegressor(min_samples_leaf=5, n_jobs=-1).fit(X_train_r.values, y_train_r)

    # Load RF shap values -- Takes 7 minutes to run so save and load
    # rf_explainer = shap.Explainer(rf)
    # rf_shap_values = rf_explainer(X_train_r)
    # joblib.dump(rf_shap_values, "rf_shap")
    rf_shap_values= joblib.load("rf_shap")

    shap_imp_xgb = feature_df_shap(xgb_shap_values.feature_names, xgb_shap_values)
    shap_imp_rf = feature_df_shap(rf_shap_values.feature_names, rf_shap_values)
    return shap_imp_xgb, shap_imp_rf

def top_k_performance(model, features, X_train, y_train, X_valid, y_valid, metric):
    top_k = {}
    selected_features = []
    for col in features:
        selected_features.append(col)
        model.fit(X_train[selected_features].values, y_train)
        m = metric(y_valid, model.predict(X_valid[selected_features].values)) 
        top_k[len(selected_features)] = m
    return top_k

def model_top_k(model, method_imps, X_train_r, y_train_r, X_test_r, y_test_r, metric):
    spearman = top_k_performance(model, method_imps['spearman'], X_train_r, y_train_r, X_test_r, y_test_r, metric)
    pca = top_k_performance(model, method_imps['pca'], X_train_r, y_train_r, X_test_r, y_test_r, metric)
    lin = top_k_performance(model, method_imps['lin'], X_train_r, y_train_r, X_test_r, y_test_r, metric)
    shap = top_k_performance(model, method_imps['shap'], X_train_r, y_train_r, X_test_r, y_test_r, metric)
    perm = top_k_performance(model, method_imps['perm'], X_train_r, y_train_r, X_test_r, y_test_r, metric)
    results = {"spearman": spearman,
                "pca": pca, 
                "lin": lin,
                "shap": shap,
                "perm": perm}
    return results

def plot_model_importances(models, method_imps, X_train_r, y_train_r, X_test_r, y_test_r, mean_absolute_error):
    lin_results = model_top_k(models['lr'], method_imps, X_train_r, y_train_r, X_test_r, y_test_r, mean_absolute_error)
    rf_results = model_top_k(models['rf'], method_imps, X_train_r, y_train_r, X_test_r, y_test_r, mean_absolute_error)
    xgb_results = model_top_k(models['xgb'], method_imps, X_train_r, y_train_r, X_test_r, y_test_r, mean_absolute_error)

    fig, axes = plt.subplots(figsize=(18, 4), nrows=1, ncols=3)
    axes = axes.flatten()
    axes[0].plot(lin_results['spearman'].keys(), lin_results['spearman'].values(), label='Spearman')
    axes[0].plot(lin_results['pca'].keys(), lin_results['pca'].values(), label='PCA')
    axes[0].plot(lin_results['lin'].keys(), lin_results['lin'].values(), label='OLS')
    axes[0].plot(lin_results['shap'].keys(), lin_results['shap'].values(), label='SHAP')
    axes[0].plot(lin_results['perm'].keys(), lin_results['perm'].values(), label='Permutation')
    axes[0].set_xlabel("Top K Features")
    axes[0].set_ylabel("MAE (Valid Set)")
    axes[0].set_title("OLS Model")
    axes[0].legend(title="Importance Strategy")

    axes[1].plot(rf_results['spearman'].keys(), rf_results['spearman'].values(), label='Spearman')
    axes[1].plot(rf_results['pca'].keys(), rf_results['pca'].values(), label='PCA')
    axes[1].plot(rf_results['lin'].keys(), rf_results['lin'].values(), label='OLS')
    axes[1].plot(rf_results['shap'].keys(), rf_results['shap'].values(), label='SHAP')
    axes[1].plot(rf_results['perm'].keys(), rf_results['perm'].values(), label='Permutation')
    axes[1].set_xlabel("Top K Features")
    axes[1].set_ylabel("MAE (Valid Set)")
    axes[1].set_title("RF Model")
    axes[1].legend(title="Importance Strategy")

    axes[2].plot(xgb_results['spearman'].keys(), xgb_results['spearman'].values(), label='Spearman')
    axes[2].plot(xgb_results['pca'].keys(), xgb_results['pca'].values(), label='PCA')
    axes[2].plot(xgb_results['lin'].keys(), xgb_results['lin'].values(), label='OLS')
    axes[2].plot(xgb_results['shap'].keys(), xgb_results['shap'].values(), label='SHAP')
    axes[2].plot(xgb_results['perm'].keys(), xgb_results['perm'].values(), label='Permutation')
    axes[2].set_xlabel("Top K Features")
    axes[2].set_ylabel("MAE (Valid Set)")
    axes[2].set_title("XGBoost Model")
    axes[2].legend(title="Importance Strategy")
    plt.show()

def importance_variance(model, X_train, y_train, X_test, y_test, metric, niter=100):
    importance_list = []
    n = len(X_train)
    for _ in range(0, niter):
        # Bootstrap training data
        idx = np.random.randint(0, n, size=n)
        X_train_ = X_train.iloc[idx]
        y_train_ = y_train.iloc[idx]
        model.fit(X_train_.values, y_train_)
        imp = permutation_importances(model, X_test, y_test, metric)
        importance_list.append(imp)
    importance_list = np.array(importance_list)
    means = importance_list.mean(axis=0)
    sds = importance_list.std(axis=0, ddof=1) 
    return means, sds

def hbar_err(x, y, err, xlabel, ylabel, title):
    plt.barh(x, y, xerr=err, capsize=10)
    # plt.xticks(rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.gca().invert_yaxis()
    plt.show()

# Empirical p-values
def importance_pval(model, X_train, y_train, X_test, y_test, metric, niter=80):
    importance_list = []
    n = len(X_train)
    for _ in range(0, niter):
        idx = np.random.randint(0, n, size=n)
        # Permute y 
        y_train_ = y_train.iloc[idx]
        model.fit(X_train.values, y_train_)
        imp = permutation_importances(model, X_test, y_test, metric)
        importance_list.append(imp)
    importance_list = np.array(importance_list)
    return importance_list

def get_p_val(null_importances, real_importances, feature_idx):
    # check both tails - extreme values
    pval = min(sum(null_importances[:,feature_idx] <= real_importances[feature_idx])/len(null_importances[:,feature_idx]), 
               sum(null_importances[:,feature_idx] >= real_importances[feature_idx])/len(null_importances[:,feature_idx])) 
    return pval

def plot_null_dist(null_importances, real_importances, feature1_idx, feature2_idx, feature_names):
    feature1_name = feature_names[feature1_idx]
    feature2_name = feature_names[feature2_idx]

    fig, axes = plt.subplots(figsize=(14,4), nrows=1, ncols=2)
    axes = axes.flatten()
    a = axes[0].hist(null_importances[:,feature1_idx], bins=10, label='Null Distribution')
    axes[0].vlines(real_importances[feature1_idx], color='r', ymin=0, ymax=np.max(a[0]), linewidth=3, label='Actual Importance')
    axes[0].set_title(f'{feature1_name} Feature')
    axes[0].set_xlabel('R2 Score Drop')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    a = axes[1].hist(null_importances[:,feature2_idx], bins=10, label='Null Distribution')
    axes[1].vlines(real_importances[feature2_idx], color='r', ymin=0, ymax=np.max(a[0]), linewidth=3, label='Actual Importance')
    axes[1].set_title(f'{feature2_name} Feature')
    axes[1].set_xlabel('R2 Score Drop')
    axes[1].set_ylabel('Count')
    axes[1].legend(bbox_to_anchor=(0.93, 1))
    fig.tight_layout()