data {
  int<lower=1> N;
  int<lower=1> T;              // Number of trials
  array[N] int<lower=1, upper=T> Tsubj;
  array[N, T] real gamble;
  array[N, T] real winLose;
  array[N, T] real happy;
  int<lower=1> happy_num;
  array[N * happy_num] int i_vec;
  array[N * happy_num] int t_vec;
}

parameters {
  vector[5] mu_pr;
  vector<lower=0>[5] sigma;
  vector[N] w0_pr;
  vector[N] w1_pr;
  vector[N] w2_pr;
  vector[N] gam_pr;
  vector[N] alp_pr;

  // vector[N] w0;
  // vector<lower=0, upper=1>[N] gam; // gamma
  // vector<lower=0, upper=1>[N] alpha;  // learning rate of phat
}

transformed parameters {
  vector[N] w0;
  vector[N] w1; // w_phat
  vector[N] w2; // w_ppe
  vector<lower=0, upper=1>[N] gam; // gamma
  vector<lower=0, upper=1>[N] alpha;  // learning rate of phat

  w0 = mu_pr[1] + sigma[1] * w0_pr;
  w1 = mu_pr[2] + sigma[2] * w1_pr;
  w2 = mu_pr[3] + sigma[3] * w2_pr;

  for (i in 1:N) {
    gam[i] = Phi_approx(mu_pr[4] + sigma[4] * gam_pr[i]);
  }
  
  for (i in 1:N) {
    alpha[i] = inv_logit(mu_pr[5] + sigma[5] * alp_pr[i]);
  }
}

model {
  // Hyperpriors
  mu_pr ~ normal(0, 1);
  // sigma ~ cauchy(0, 2.5);
  sigma ~ normal(0, 1) T[0, ];
  // sigma ~ normal(0, 0.2)T[0, ];

  // Individual-level priors
  w0_pr ~ student_t(3, 0, 1);      // outlier 포함 가능
  w1_pr ~ student_t(3, 0, 1);      // outlier 포함 가능
  w2_pr ~ student_t(3, 0, 1);      // outlier 포함 가능
  gam_pr ~ student_t(3, 0, 1);      // outlier 포함 가능
  alp_pr ~ student_t(3, 0, 1);      // outlier 포함 가능

  // partial individual
  // w0 ~ student_t(3, 0, 1); 
  // gam ~ beta(2, 2); 
  // alpha ~ beta(2, 2); 

  //w0_pr ~ normal(0, 1);
  //w1_pr ~ normal(0, 1);
  //w2_pr ~ normal(0, 1);
  //gam_pr ~ normal(0, 1);
  //alp_pr ~ normal(0, 1);

  int count = 1;

  for (i in 1:N) {
    real Phat = 0.5;
    real Phat_decay = 0;      
    real PPE_decay = 0;         

    // EDITED: Since most recent value should not be Decayed
    for (t in 1:Tsubj[i]) {
      if(gamble[i, t] == 0){ // did not gamble
        PPE_decay = gam[i] * PPE_decay;
        Phat_decay = gam[i] * Phat_decay;
      } else {
        PPE_decay = gam[i] * PPE_decay + (winLose[i, t] - Phat);
        Phat_decay = gam[i] * Phat_decay + Phat;
        Phat = Phat + alpha[i] * (winLose[i, t] - Phat);
        Phat = fmin(0.999, fmax(0.001, Phat));
      }

      // EDITED: since happy is measured after trial, it should come after decaysum
      if (i == i_vec[count] && t == t_vec[count]) {
        real mu = w0[i] + w1[i] * Phat_decay + w2[i] * PPE_decay;
        happy[i, t] ~ normal(mu, 1.0);
        count = count + 1;
      }
    }
  }
}

generated quantities {
    real mu_w0;
    real mu_w1;
    real mu_w2;
    real<lower=0, upper=1> mu_gam;
    real<lower=0, upper=1> mu_alp;
    
    array[N, happy_num] real log_lik;
    array[N, happy_num] real y_pred;
    
    // Set all posterior predictions to -1 (avoids NULL values)
    for (i in 1:N) {
        for (t in 1:happy_num) {
          log_lik[i, t] = -1;
          y_pred[i, t] = -1;
        }
    }
    
    // For parameter recovery
    mu_w0    = mu_pr[1];
    mu_w1    = mu_pr[2];
    mu_w2    = mu_pr[3];
    mu_gam   = Phi_approx(mu_pr[4]);
    mu_alp   = Phi_approx(mu_pr[5]);

    // as a local section
    { 
    int count = 1;
    
    for (i in 1:N) {
        real Phat = 0.5;
        real Phat_decay = 0;
        real PPE_decay = 0;         // 
        int t_count = 1;
    
        // EDITED: Since most recent value should not be Decayed
        for (t in 1:Tsubj[i]) {
          if(gamble[i, t] == 0){ // did not gamble
            PPE_decay = gam[i] * PPE_decay;
            Phat_decay = gam[i] * Phat_decay;
          } else {
            PPE_decay = gam[i] * PPE_decay + (winLose[i, t] - Phat);
            Phat_decay = gam[i] * Phat_decay + Phat;
            Phat = Phat + alpha[i] * (winLose[i, t] - Phat);
            Phat = fmin(0.999, fmax(0.001, Phat));
          }
    
          // EDITED: since happy is measured after trial, it should come after decaysum
          if (i == i_vec[count] && t == t_vec[count]) {
            real mu = w0[i] + w1[i] * Phat_decay + w2[i] * PPE_decay;
            y_pred[i, t_count] = normal_rng(mu, 1.0);
            log_lik[i, t_count] = normal_lpdf(happy[i, t] | mu, 1.0);  
            count = count + 1;
            t_count = t_count + 1;         
          }
        }
      }
    }
}
