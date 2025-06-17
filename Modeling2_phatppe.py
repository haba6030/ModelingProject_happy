from cmdstanpy import install_cmdstan
import os
from scipy.stats import norm
from cmdstanpy import CmdStanModel
import random
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import json
from tqdm import tqdm


# # Import data
dfBlInfo = pd.read_csv("combined_participant_info.csv")
dfBlTrial = pd.read_csv("Blain_trials_modified.csv")

# # Modeling 2 - Introducing PPE(Blain, 2020)
# Include PPE
# ## Phat + PPE model

df = dfBlTrial
cols_to_fix = ["mag1", "mag2","prob1", "winLose", "happiness"]

for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ### MLE 모델 제작
# 2. Overall model : happiness = phat + ppe
def predict_happiness(params, Winlose, chose_risky, choice, rated_mask):
    w0, w_phat, w_PPE, gamma, alpha = params
    T = len(Winlose)

    # 초기화
    phat = 0.5
    PPE = 0
    phat_decay = np.zeros(T)
    PPE_decay = np.zeros(T)

    if chose_risky[0]:           
        phat_decay[0] = phat
        PPE = Winlose[0] - phat
        PPE_decay[0] = PPE
        phat = phat + alpha * PPE
        phat = min(0.999, max(0.001, phat))
    
    # 순차 업데이트
    for t in range(1, T):
        if chose_risky[t]: # gambled                
            PPE = Winlose[t] - phat
            PPE_decay[t] = gamma * PPE_decay[t-1] + PPE
            phat_decay[t] = gamma * phat_decay[t-1] + phat
            phat = phat + alpha * PPE
            phat = min(0.999, max(0.001, phat))
        else: # chose certain: gamble value only decay
            #phat = phat  # winning percentage will not change
            #ppe = ppe
            phat_decay[t] = gamma * phat_decay[t-1] # but effect will decrease
            PPE_decay[t] = gamma * PPE_decay[t-1]

    mu = w0 + w_phat * phat_decay + w_PPE * PPE_decay
    return mu





# 📌 3. Negative loglikelihood
def nll_rated(params, Winlose, chose_risky, choice, rated_mask, H_obs):
    w0, w_phat, w_PPE, gamma, alpha = params
    if not (0 <= gamma <= 1 and 0 <= alpha <= 1):
        return np.inf

    mu = predict_happiness(params, Winlose, chose_risky, choice, rated_mask)
    mu_rated = mu[rated_mask]

    negloglik = -np.sum(norm.logpdf(H_obs, loc=mu_rated, scale=1.0))  # fixed sigma
    return negloglik

# 4. Evaluate model
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

# 참가자별 MLE
def fit_mle_per_participant(df_trials, id_x = "id", happiness = "happiness", 
                            choseRisky_idx = "choseRisky", choice_idx = "choice",
                            winlose_col = "winLose"):
    results = []
    eval_list = []
    
    for pid, group in df_trials.groupby(id_x):
        group = group.reset_index(drop=True)

        # 루프가 돌아갈 수 없는 경우
        if group[happiness].notna().sum() < 5: 
            continue

        # 변수 생성
        Winlose = group[winlose_col].to_numpy()
        chose_risky = group[choseRisky_idx]
        choice = group[choice_idx]
        rated_mask = group[happiness].notna().values
        H_obs = group.loc[rated_mask, happiness].values

        # 최적화
        best_loss = np.inf
        best_params = None
        
        for _ in range(10):  # 10번 반복
            # w0, wphat, wppe, gamma, alpha
            x0 = np.random.uniform(0, 1, size=5)
            x0[3] = np.random.uniform(0.1, 0.95)  # gamma는 0~1 사이에서 안전하게
            x0[4] = np.random.uniform(0.1, 0.95)  # alpha도
        
            res = minimize(
                nll_rated, 
                x0=x0, 
                args=(Winlose, chose_risky, choice, rated_mask, H_obs),
                bounds=[(None, None)] * 3 + [(0, 1), (0, 1)],  # gamma, alpha
                method="L-BFGS-B"
            )
        
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x

        # 예측값 생성
        logL = -best_loss
        mu_full = predict_happiness(best_params, Winlose, chose_risky, choice, rated_mask)
        y_true = H_obs
        y_pred = mu_full[rated_mask]
    
        metrics = evaluate_mle(y_true, y_pred, logL, k_params=len(best_params))
        metrics["participant_id"] = pid
        eval_list.append(metrics)
                
        # 최적화 결과 저장
        results.append([pid] + list(best_params))

    df_result = pd.DataFrame(results, columns=["id", "w0", "wPhat", "w_PPE", "gamma", "alpha"])
    df_eval = pd.DataFrame(eval_list)
    return df_result, df_eval

print("MLE")
df = dfBlTrial
df_params, df_eval = fit_mle_per_participant(df)

# # 결과 저장
df_params.to_csv("mle_bla_phat_results.csv", index=False)
df_eval.to_csv("mle_bla_phat_evals.csv", index=False)


# ## Individual Bayesian
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
    #y_pred_path = f"outputs/{prefix}_posterior_y_pred.csv"
    #np.savetxt(y_pred_path, y_pred_flat, delimiter=",")

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

participants = df["id"].unique()
N = len(participants)
T = df.groupby("id").size().max()  # 160으로 동일

Tsubj = df.groupby("id").size().values.astype(int)
gamble = to_matrix(df, "choseRisky")
winLose = to_matrix(df, "winLose")
happy = to_matrix(df, "happiness", fill=-999)  

# Stan 데이터 구성
stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj.tolist(),
    "gamble": gamble.tolist(),
    "winLose": winLose.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}

with open("bldata_phatppe.json", "w") as f:
    json.dump(stan_data, f)

model = CmdStanModel(stan_file='happy_phatppe_indi.stan')
fit = model.sample(data="bldata_phatppe.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)
results = save_stan_outputs_and_evaluation(fit, prefix="Ind_phatppe_bla_2nd")
print(results["eval_csv"])  # 저장된 평가 지표 파일 경로


# ## Hierarchical Bayesian
model = CmdStanModel(stan_file='happy_phatppe_hba.stan')
fit = model.sample(data="bldata_phatppe.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)

results = save_stan_outputs_and_evaluation(fit, prefix="HBA_phatppe_bla_2nd")







