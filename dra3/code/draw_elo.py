import json, tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

BASELINE_NAME = "baseline"
BASELINE_ELO = 1000.0

def calc_elo(data:list[dict]):
    players = sorted({d[k] for d in data for k in ("p1", "p2") if k in d} - {BASELINE_NAME})
    player2idx = {p: i for i, p in enumerate(players)}
    num_players = len(players)
    
    elo_scores = nn.Parameter(torch.zeros(num_players)) 
    first_move_bias = nn.Parameter(torch.tensor(0.0))
    
    optimizer = optim.Adam([elo_scores, first_move_bias], lr=20)
    
    T = tqdm.trange(200)
    
    for epoch in T:
        all_loss = []
        for item in data:
            r_b = torch.tensor(BASELINE_ELO) if item["p1"] == BASELINE_NAME else elo_scores[player2idx[item["p1"]]]
            r_w = torch.tensor(BASELINE_ELO) if item["p2"] == BASELINE_NAME else elo_scores[player2idx[item["p2"]]]
            
            delta = (r_b + first_move_bias) - r_w
            
            prob_black_win = 1 / (1 + 10 ** (-delta / 400))
            prob_white_win = 1 - prob_black_win
            prob_draw = torch.clamp(1 - prob_black_win - prob_white_win, min=1e-6)
            
            if item["p1_win"] > 0: all_loss.append(-torch.log(prob_black_win + 1e-6)*item["p1_win"])
            if item["p2_win"] > 0: all_loss.append(-torch.log(prob_white_win + 1e-6)*item["p2_win"])
            if item["draw"] > 0: all_loss.append(-torch.log(prob_draw + 1e-6)*item["draw"])
        
        loss = torch.stack(all_loss).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        T.set_description_str(f"Loss: {loss.detach().cpu().item():4f}")
    return players, elo_scores.cpu().tolist()

def draw(file_path:str, save_path:str="elo.png"):
    with open(file_path, "r") as f:
        d = json.load(f)
    players, elo_scores = calc_elo(d)
    steps = [int(int(''.join(filter(str.isdigit, player)))) for player in players]
    plt.plot(steps, elo_scores)
    plt.savefig(save_path)
    print(f"elo curve saved to {save_path}")


if __name__ == "__main__":
    draw("eval_results.json")