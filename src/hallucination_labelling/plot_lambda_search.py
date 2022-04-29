import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import generate_lambdas
from pylab import cm

def extract_ROUGE_L_F1(filename):
    with open(filename, 'r') as f:
        line = f.readlines()[13]
        return float(line.split()[3])

def extract_probs(filename):
    with open(filename, 'r') as f:
        line = f.readlines()[0].split()
        return float(line[0]), float(line[1]), float(line[2]),

def extract_data(log_dir_rouge, log_dir_probs):
    lambdas = generate_lambdas.main()

    data = {"lambdas": lambdas, "scores":[], "probs":[], "hall_probs":[], "first_hall_probs":[]}
    for lamb in lambdas:
        filename =  log_dir_rouge + "score_%0.4E" % lamb
        score = extract_ROUGE_L_F1(filename)
        data["scores"].append(score)

        filename =  log_dir_probs + "avg_probs_%0.4E" % lamb
        probs, hall_probs, first_hall_probs = extract_probs(filename)
        data["probs"].append(probs)
        data["hall_probs"].append(hall_probs)
        data["first_hall_probs"].append(first_hall_probs)
    data["probs"] = np.array(data["probs"])
    data["hall_probs"] = np.array(data["hall_probs"])
    data["first_hall_probs"] = np.array(data["first_hall_probs"])

    return data

def plot(prob_label, filename=None, show_fig = True, save_fig = False):
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    c1 = "red"
    ax1.set_xlabel(r'$\mathregular{\lambda}$', labelpad=10)
    ax1.set_ylabel(r'ROUGE-L $\mathregular{F_1}$ score', labelpad=10, color=c1)
    ax1.plot(data["lambdas"], data["scores"], color=c1, marker='o', linewidth=2, label='ROUGE')
    ax1.tick_params(axis='y', labelcolor=c1)

    c2 = "blue"
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Average token log probabilities', labelpad=10, color=c2)
    ax2.plot(data["lambdas"], data[prob_label], color=c2, marker='o', linewidth=2, label='Log probabilities')
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    if show_fig:
        plt.show()

    if save_fig:
        path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/"
        plt.savefig(path + filename, dpi=150, bbox_inches = "tight")


if __name__ == '__main__':
    log_dir_rouge = "logs/hallucination_labelling/11_full/"
    log_dir_probs = "logs/hallucination_labelling/11_ref_full/"

    data = extract_data(log_dir_rouge, log_dir_probs)
    plot("probs", filename="hall_labelled_lambda_search_all_tokens.png", show_fig=False, save_fig=True)
    plot("hall_probs", filename="hall_labelled_lambda_search_hall.png", show_fig=False, save_fig=True)
    plot("first_hall_probs", filename="hall_labelled_lambda_search_first_hall.png", show_fig=False, save_fig=True)
