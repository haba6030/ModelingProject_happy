#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cmdstanpy import install_cmdstan
install_cmdstan()


# In[2]:


from cmdstanpy import CmdStanModel


# In[3]:


import os
from scipy.stats import norm


# In[4]:


import random
import numpy as np
import pandas as pd
import arviz as az
print(az.__version__)

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import json
from tqdm import tqdm


# # Import data

# In[7]:


dfBlInfo = pd.read_csv("combined_participant_info.csv")
dfBlTrial = pd.read_csv("Blain_trials_modified.csv")


# In[8]:


dfBlTrial['choseRisky'] = 1.0


# In[9]:


# dfRtTrial = pd.read_csv("Rutledge_trials_modified.csv")
# dfRtInfo = pd.read_csv("Rutledge_info_modified.csv")


# In[10]:


print(dfBlInfo.shape)
print(dfBlTrial.shape)
# print(dfRtInfo.shape)
# print(dfRtTrial.shape)


# In[11]:


dfBlTrial


# In[12]:


# # Fortest
# print("For test")
# dfBlInfo = dfBlInfo[:10]
# dfBlTrial = dfBlTrial[:1600]


# In[13]:


# print(dfBlInfo.shape)
# print(dfBlTrial.shape)


# # Modeling 1 - Original Happiness Model
# Apply original Stan model in data

# ## MLE

# In[51]:


# 📌 2. Overall model
def predict_happiness(params, CR, EV, RPE, chose_risky):
    w0, w_CR, w_EV, w_RPE, gamma = params
    T = len(CR)

    # 초기화
    CR_decay = np.zeros(T)
    EV_decay = np.zeros(T)
    RPE_decay = np.zeros(T)

    if chose_risky[0]:
        EV_decay[0] = EV[0]
        RPE_decay[0] = RPE[0]
    else:
        CR_decay[0] = CR[0]
    
    # 순차 업데이트
    for t in range(1, T):
        if chose_risky[t]: # gambled
            CR_decay[t] = gamma * CR_decay[t-1]
            EV_decay[t] = gamma * EV_decay[t-1] + EV[t]
            RPE_decay[t] = gamma * RPE_decay[t-1] + RPE[t]
        else: # chose certain: gamble value only decay
            CR_decay[t] = gamma * CR_decay[t-1] + CR[t]
            EV_decay[t] = gamma * EV_decay[t-1]
            RPE_decay[t] = gamma * RPE_decay[t-1]

    mu = w0 + w_CR * CR_decay + w_EV * EV_decay + w_RPE * RPE_decay
    return mu


# In[52]:


# 📌 3. Negative loglikelihood
def nll_rated(params, CR, EV, RPE, chose_risky, rated_mask, H_obs):
    w0, w_CR, w_EV, w_RPE, gamma = params

    mu = predict_happiness(params, CR, EV, RPE, chose_risky)
    mu_rated = mu[rated_mask]

    negloglik = -np.sum(norm.logpdf(H_obs, loc=mu_rated, scale=1.0))  # fixed sigma
    return negloglik


# In[53]:


from sklearn.metrics import mean_squared_error, r2_score

# 📌 4. Evaluate model
def evaluate_mle(y_true, y_pred, log_likelihood, k_params):
    n = len(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    aic = 2 * k_params - 2 * log_likelihood
    bic = k_params * np.log(n) - 2 * log_likelihood
    return {
        "log_likelihood": log_likelihood,
        "rmse": rmse,
        "r2": r2,
        "aic": aic,
        "bic": bic
    }


# In[54]:


# 참가자별 MLE
def fit_mle_per_participant(df_trials, id_x = "id", happiness = "happiness", 
                            choseRisky_idx = "choseRisky", outcome_idx = "outcome",
                            CR_idx = "CR", winValue = "mag1_norm", loseValue = "mag2_norm", 
                            prob1 = "prob1" , prob2 = "prob2"):
    results = []
    eval_list = []
    
    for pid, group in df_trials.groupby(id_x):
        group = group.reset_index(drop=True)

        # 루프가 돌아갈 수 없는 경우
        if group[happiness].notna().sum() < 5: 
            continue

        # 변수 생성
        CR = group[CR_idx].fillna(0).values
        EV = (group[prob1] * group[winValue] + group[prob2] * group[loseValue]).fillna(0).values
        RPE = group[outcome_idx] - EV
        choseRisky = group[choseRisky_idx]
        rated_mask = group[happiness].notna().values
        H_obs = group.loc[rated_mask, happiness].values

        # 최적화
        best_loss = np.inf
        best_params = None
        
        for _ in range(10):  # 10번 반복
            x0 = np.random.uniform(0, 1, size=5)
            x0[-1] = np.random.uniform(0.1, 0.95)  # gamma는 0~1 사이에서 안전하게
        
            res = minimize(
                nll_rated, 
                x0=x0, 
                args=(CR, EV, RPE, choseRisky, rated_mask, H_obs),
                bounds=[(None, None)]*4 + [(0.0, 1.0)],
                method="L-BFGS-B"
            )
        
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x

        # 예측값 생성
        logL = -best_loss
        mu_full = predict_happiness(best_params, CR, EV, RPE, choseRisky)
        y_true = H_obs
        y_pred = mu_full[rated_mask]
    
        metrics = evaluate_mle(y_true, y_pred, logL, k_params=len(best_params))
        metrics["participant_id"] = pid
        eval_list.append(metrics)
                
        # 최적화 결과 저장
        results.append([pid] + list(best_params))

    df_result = pd.DataFrame(results, columns=["id", "w0", "w_CR", "w_EV", "w_RPE", "gamma"])
    df_eval = pd.DataFrame(eval_list)
    return df_result, df_eval


# In[55]:


# #Rutledge
# # fit_mle_per_participant(df_trials, id_x, happiness_z, CR, winValue, loseValue, prob1, prob2):

# df = dfRtTrial 
# df_params_Rt, df_eval_Rt = fit_mle_per_participant(df)


# In[56]:


# print(df_params_Rt)
# print(df_eval_Rt)


# In[1]:


# df = dfBlTrial
# print("MLE")
# df_params_Bl, df_eval_Bl = fit_mle_per_participant(df)


# In[17]:


# df_params_Bl.to_csv("mle_bla_1st_raw_results.csv", index=False)
# df_eval_Bl.to_csv("mle_bla_1st_raw_evals.csv", index=False)


# ## Individual Hierarchical

# In[27]:


# Matrix 함수: for json
def to_matrix(df, varname, fill=0):
    mat = np.full((N, T), fill, dtype=np.float32)
    for i, pid in enumerate(participants):
        values = df[df["id"] == pid][varname].fillna(fill).values 
        mat[i, :len(values)] = values
    return mat

def to_int_matrix(df, varname, fill=0):
    mat = np.full((N, T), fill, dtype=np.int32)
    for i, pid in enumerate(participants):
        values = df[df["id"] == pid][varname].fillna(fill).astype(int).values 
        mat[i, :len(values)] = values
    return mat


# In[28]:


# df = dfRtTrial

# participants = df["id"].unique()
# N = len(participants)
# T = df.groupby("id").size().max()  # 160으로 동일

# Tsubj = df.groupby("id").size().values.astype(int)
# gain = to_matrix(df, "mag1")
# loss = to_matrix(df, "mag2")
# prob1 = to_matrix(df, "prob1")
# prob2 = to_matrix(df, "prob2")
# cert = to_matrix(df, "CR")
# gamble = to_matrix(df, "choseRisky")
# outcome = to_matrix(df, "outcome")
# happy = to_matrix(df, "happiness", fill=-999)  

# # Stan 데이터 구성
# stan_data = {
#     "N": N,
#     "T": T,
#     "Tsubj": Tsubj.tolist(),
#     "gain": gain.tolist(),
#     "loss": loss.tolist(),
#     "prob1": prob1.tolist(),
#     "prob2": prob2.tolist(),
#     "cert": cert.tolist(),
#     "gamble": cert.tolist(),
#     "outcome": outcome.tolist(),
#     "happy": happy.tolist(),
# }

# with open("rtdata_1st.json", "w") as f:
#     json.dump(stan_data, f)


# In[50]:


df = dfBlTrial 
# print(df.groupby("id").size().to_string())
print(df["id"].unique().shape)


# In[2]:


print(df)


# In[31]:


# happiness 관측값 위치 확인
df = dfBlTrial

i_vec = []
t_vec = []
T_total = 0
T_temp = 0
T_max = 0
T_list = []

participants = df["id"].unique()
N = len(participants)
T = df.groupby("id").size().max()  # 160으로 동일

for i in range(N):
    T_temp = 0
    for t in range(T):
        if not pd.isna(df["happiness"].iloc[i * T + t]):
            i_vec.append(i + 1) #stan 기준
            t_vec.append(t + 1)
            T_total += 1
            T_temp += 1
    T_list.append(T_temp)
    if T_temp > T_max:
            T_max = T_temp


# In[32]:


df = dfBlTrial
df["gamble"] = 1

participants = df["id"].unique()
N = len(participants)
T = df.groupby("id").size().max()  # 160으로 동일
print(T)

Tsubj = df.groupby("id").size().values.astype(int)
gain = to_matrix(df, "mag1_norm")
loss = to_matrix(df, "mag2_norm")
prob1 = to_matrix(df, "prob1")
prob2 = to_matrix(df, "prob2")
cert = to_matrix(df, "CR")
gamble = to_matrix(df, "gamble")
outcome = to_matrix(df, "outcome")
happy = to_matrix(df, "happiness", fill=-999)


# Stan 데이터 구성
stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj.tolist(),
    "gain": gain.tolist(),
    "loss": loss.tolist(),
    "prob1": prob1.tolist(),
    "prob2": prob2.tolist(),
    "cert": cert.tolist(),
    "gamble": cert.tolist(),
    "outcome": outcome.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}
with open("bldata_1st.json", "w") as f:
    json.dump(stan_data, f)


# In[33]:


def save_stan_outputs_and_evaluation(fit, prefix="model", trace_chunk_size=20):
    # 0. Create output directory
    os.makedirs("outputs", exist_ok=True)

    # 1. Summary of parameters
    summary_df = fit.summary()
    summary_path = f"outputs/{prefix}_parameter_summary.csv"
    summary_df.to_csv(summary_path)

    # 2. Posterior predicted values (y_pred)
    y_pred = fit.stan_variable("y_pred")
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    # y_pred_path = f"outputs/{prefix}_posterior_y_pred.csv"
    # np.savetxt(y_pred_path, y_pred_flat, delimiter=",")

    # 3. Log-likelihood samples
    log_lik = fit.stan_variable("log_lik").copy()
    log_lik[log_lik == -1] = np.nan
    log_lik = log_lik.astype(np.float64)  # ensure proper dtype

    log_lik_mean = np.nanmean(log_lik, axis=0)  # shape: (N, T)
    log_lik_path = f"outputs/{prefix}_log_lik.csv"
    np.savetxt(log_lik_path, log_lik_mean, delimiter=",")
 
    # 4. ArviZ evaluation
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive="y_pred",
        log_likelihood="log_lik"
    )

    waic = az.waic(idata)
    loo = az.loo(idata)
    
    # Pareto-k diagnostics CSV and plot
    pareto_k_vals = loo.pareto_k.values.flatten()
    pareto_k_df = pd.DataFrame({"pareto_k": pareto_k_vals})
    pareto_k_df["observation"] = np.arange(len(pareto_k_vals))
    pareto_k_path = f"outputs/{prefix}_pareto_k.csv"
    pareto_k_df.to_csv(pareto_k_path, index=False)

    plt.figure(figsize=(10, 4))
    az.plot_khat(loo)
    plt.title("Pareto-k Diagnostic (per observation)")
    plt.tight_layout()
    pareto_k_plot_path = f"outputs/{prefix}_pareto_k_plot.png"
    plt.savefig(pareto_k_plot_path, dpi=300)
    plt.close()

    # AIC and BIC using log-likelihood
    log_lik_total = np.sum(log_lik, axis=-1)  # sum over trials
    log_lik_mean = np.mean(log_lik_total)
    n = np.prod(log_lik.shape[1:])  # total number of likelihood points
    k = summary_df.shape[0]  # number of parameters
    aic = 2 * k - 2 * log_lik_mean
    bic = k * np.log(n) - 2 * log_lik_mean

    eval_df = pd.DataFrame({
        "elpd_waic": [waic.elpd_waic],
        "waic_se": [waic.se],
        "elpd_loo": [loo.elpd_loo],
        "loo_se": [loo.se],
        "aic": [aic],
        "bic": [bic],
        "log_lik_total_mean": [log_lik_mean]
    })
    eval_path = f"outputs/{prefix}_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)

    # 5. Generated quantities
    stan_vars = fit.stan_variables().keys()
    gen_quant_vars = [var for var in stan_vars if var.startswith("mu_")]
    gen_quant_stats = {}

    for var in gen_quant_vars:
        print(var)
        if var in fit.stan_variables():
            data = fit.stan_variable(var)
            gen_quant_stats[var] = {
                "mean": np.mean(data),
                "std": np.std(data),
                "median": np.median(data),
                "min": np.min(data),
                "max": np.max(data)
            }
            # Plot histogram
            plt.hist(data, bins=30, alpha=0.7)
            plt.title(f"Posterior of {var}")
            plt.xlabel(var)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"outputs/{prefix}_posterior_{var}.png", dpi=300)
            plt.close()

    gen_quant_df = pd.DataFrame(gen_quant_stats).T
    gen_quant_csv = f"outputs/{prefix}_generated_quantities_summary.csv"
    gen_quant_df.to_csv(gen_quant_csv)

    # 8. Traceplot (chunked if too many parameters)
    rhat_mean = summary_df["R_hat"].mean()
    all_params = list(idata.posterior.data_vars.keys())
    traceplot_paths = []

    for i in range(0, len(all_params), trace_chunk_size):
        subset = all_params[i:i+trace_chunk_size]
        az.plot_trace(idata, var_names=subset, compact=True)
        fig = plt.gcf()  # 현재 활성화된 figure 가져오기
        fig.suptitle(f"Trace Plot (Rhat ≈ {rhat_mean:.2f})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = f"outputs/{prefix}_traceplot_part{i//trace_chunk_size+1}.png"
        plt.savefig(path, dpi=300)
        traceplot_paths.append(path)
        plt.close()

    return {
        "summary_csv": summary_path,
        #"y_pred_csv": y_pred_path,
        "log_lik_csv": log_lik_path,
        "eval_csv": eval_path,
        "traceplot_pngs": traceplot_paths,
        "waic": waic.elpd_waic,
        "loo": loo.elpd_loo,
        "aic": aic,
        "bic": bic
    }


# ### 예측 시작

# #### Blain

# In[50]:


model = CmdStanModel(stan_file='happy_2014_indiv.stan')


# In[51]:


fit = model.sample(data="bldata_1st.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)


# In[52]:


results = save_stan_outputs_and_evaluation(fit, prefix="Ind_raw_bla_1st")
print(results["eval_csv"])  # 저장된 평가 지표 파일 경로


# ## HBA

# ### Stan Compile

# In[ ]:


# model = CmdStanModel(stan_file='happy_2014.stan')


# In[ ]:


# fit = model.sample(data="rtdata_1st.json", chains=4, parallel_chains=4, 
#                    iter_warmup=1000, iter_sampling=1000, 
#                    seed=2025, adapt_delta=0.95,
#                    show_progress=True)


# In[ ]:


# results = save_stan_outputs_and_evaluation(fit, prefix="HBA_raw_rut_1st")
# print(results["eval_csv"])  # 저장된 평가 지표 파일 경로


# In[4]:


model = CmdStanModel(stan_file='happy_2014.stan')


# In[46]:


fit = model.sample(data="bldata_1st.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)


# In[47]:


results = save_stan_outputs_and_evaluation(fit, prefix="HBA_raw_bla_1st")
print(results["eval_csv"])  # 저장된 평가 지표 파일 경로


# In[ ]:


# # 6. 결과 요약 summary → CSV 저장
# summary = fit.summary()
# summary.to_csv("HBA_bla_1st_parameter_summary.csv")

# # 8. posterior 예측값 y_pred → npy 또는 csv 저장
# y_pred = fit.stan_variable("y_pred")
# np.savetxt("HBA_bla_1st_posterior_y_pred.csv", y_pred.reshape(y_pred.shape[0], -1), delimiter=",")  # 구조 변형 후 csv

# # 9. log-likelihood → CSV 또는 npz
# log_lik = fit.stan_variable("log_lik")
# np.savetxt("HBA_bla_1st_log_lik.csv", log_lik, delimiter=",")


# ### Parameter Recovery

# In[ ]:


# 논문 질문 gpt 참고


# ## CBM
