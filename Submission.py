from sys import exit, exc_info, argv
import numpy as np
import pandas as pd

from netsapi.challenge import *

def time_diff_ave(ar):
    k = 0
    for i in range(len(ar)-1):
        k += abs(ar[str(i+2)][0] - ar[str(i+1)][0])
        k += abs(ar[str(i+2)][1] - ar[str(i+1)][1])
    return k / (len(ar) - 1)

def first_year_first(ar):
    return ar[str(1)][0]

def first_ave(ar):
    k = 0
    for i in range(len(ar)-1):
        k += ar[str(i+1)][0]
    return k / (len(ar) - 1)

def second_ave(ar):
    k = 0
    for i in range(len(ar)-1):
        k += ar[str(i+1)][1]
    return k / (len(ar) - 1)

def ave_diff(ar):
    x = 0
    for i in range(len(ar)):
        x += (ar[str(i+1)][0] - ar[str(i+1)][1])

    return x / len(ar)

def ave_sum(ar):
    x = 0
    for i in range(len(ar)):
        x += (ar[str(i+1)][0] + ar[str(i+1)][1])

    return x / len(ar)

def pair_diff_ave(ar):
    a = []
    for i in range(len(ar)):
        a.append(float(abs(ar[str(i+1)][0] - ar[str(i+1)][1])))

    return np.average(a)

def pair_sum_ave(ar):
    a = []
    for i in range(len(ar)):
        a.append(float(ar[str(i+1)][0] + ar[str(i+1)][1]))

    return np.average(a)

def pair_diff_var(ar):
    a = []
    for i in range(len(ar)):
        a.append(float(ar[str(i+1)][0] - ar[str(i+1)][1]))
    return np.var(a)

def pair_sum_var(ar):
    a = []
    for i in range(len(ar)):
        a.append(float(ar[str(i+1)][0] + ar[str(i+1)][1]))
    return np.var(a)

def calc_v(r, c, func, p):
    l = list(map(func, c))
    cor = np.corrcoef(r, l)[0,1]
    max_v = max(l)
    min_v = min(l)
    ret = 0.0
    if max_v > min_v:
        if cor >= 0.0:
            ret = (func(p) - min_v) / (max_v - min_v) * (np.exp(abs(cor)) - 1.0)
        else:
            ret = - (func(p) - min_v) / (max_v - min_v) * (np.exp(abs(cor)) - 1.0)
    return ret

class CustomAgent:
    funcs = {
            'time_diff_ave':time_diff_ave, 
            'first_year_first':first_year_first,
            'pair_sum_ave':pair_sum_ave, 'pair_diff_ave':pair_diff_ave,
            'pair_sum_log_var':pair_sum_var, 'pair_diff_log_var': pair_diff_var,
            'ave_diff':ave_diff, 'ave_sum':ave_sum, 'first_ave': first_ave, 'second_ave':second_ave }

    def __init__(self, environment):
        self.environment = environment
    
    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        
        rewards = []

        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            total_count = 20
            ph1_count = 2
    
            for i in range(total_count):
                self.environment.reset()
                policy = {}
                if i < ph1_count:
               
                    for j in range(5): #episode length
  
                        p = random.random()
                        q = random.random()
                        policy[str(j+1)]=[q, p]
                else:

                    best_v = -float('Inf')
                    best_policy_init = {}
                    for k in range(int((i + 1)**2 / 5) + 1):
                        p = random.random()
                        if k == 0 or p > 0.8:
                            p1 = random.random()
                            q1 = random.random()
                            p2 = random.random()
                            q2 = random.random()
                        elif p > 0.4:
                            p1 = random.random()
                            q1 = random.random()
                            p2 = best_policy_init[str(2)][1]
                            q2 = best_policy_init[str(2)][0]
                        else:
                            p1 = best_policy_init[str(1)][1]
                            q1 = best_policy_init[str(1)][0]
                            p2 = random.random()
                            q2 = random.random()
                        va = []
                        policy[str(1)]=[q1, p1]
                        policy[str(2)]=[q2, p2]
                        for f in self.funcs.values():
                            va.append(calc_v(rewards, candidates, f, policy))

                        v = sum(va)
                        if v > best_v:
                            best_v = v
                            best_va = va
                            best_policy_init = policy.copy()
                    policy = best_policy_init

                    best_v = -float('Inf')
                    best_va = []
                    for j in range(2,5): #episode length
                        
                        best_v = -float('Inf')
                        best_va = []
                        b_policy = []
                        for k in range(int((i + 1)**2 / 5) + 1):
                            p = random.random()
                            q = random.random()
                            va = []
                            policy[str(j+1)]=[q, p]
                            for f in self.funcs.values():
                                va.append(calc_v(rewards, candidates, f, policy))

                            v = sum(va)
                            if v > best_v:
                                best_v = v
                                best_va = va
                                b_policy = [q, p]

                        policy[str(j+1)] = b_policy

                candidates.append(policy)
                rew = self.environment.evaluatePolicy(policy)
                rewards.append( rew )

            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]
            print(best_reward, best_policy)
            
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
            
        return best_policy, best_reward

def main():
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "example.csv")


if __name__ == '__main__':
    main()
