import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

def read_lines(filename, datatype):
    lines = []
    with open(filename) as file:
        for line in file:
            l = list(map(datatype, line.rstrip().split()))
            lines.append(l)
    return lines

def get_hall(labels, entropies):
    hall = []
    non_hall = []
    labels = sum(labels, [])
    entropies = sum(entropies, [])
    for is_hal, ent in zip(labels, entropies):
        if is_hal:
            hall.append(ent)
        else:
            non_hall.append(ent)
    return hall, non_hall

def get_first_hall(labels, entropies):
    first_hall = []
    for i in range(len(labels)):
        prev_label = 0
        for is_hal, ent in zip(labels[i], entropies[i]):
            is_first = is_hal and not prev_label
            if is_first:
                first_hall.append(ent)
            prev_label = is_hal

    return first_hall

def get_subseq_hall(labels, entropies):
    subseq_hall = []
    for i in range(len(labels)):
        prev_label = 0
        for is_hal, ent in zip(labels[i], entropies[i]):
            is_subseq  = is_hal and prev_label
            if is_subseq:
                subseq_hall.append(ent)
            prev_label = is_hal
    return subseq_hall

def t_test(x, y):
    t_stat, p_val = st.ttest_ind(x, y)
    if p_val <= 0.01:
        res = "Reject"
    else:
        res = "Cannot reject"
    print((res, "%.4E"%t_stat, "%.4E"%p_val))

def ks_test(x, y):
    ks_stat, p_val = st.ks_2samp(x, y)
    if p_val <= 0.01:
        res = "Reject"
    else:
        res = "Cannot reject"
    print((res, "%.4E"%ks_stat, "%.4E"%p_val))

def get_max_val(x, y):
    return np.ceil(max(max(x), max(y)))

def distrib(x, y, saveplots, showplots, filename=None, x_name="Hallucinated", y_name="Non-Hallucinated"):
    max_val = get_max_val(x, y)
    bins = np.linspace(0, max_val, 100)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2


    fig = plt.figure(figsize=(10, 6), dpi=100)

    plt.hist(x, alpha=0.5, bins=bins, density=True, label=x_name)
    plt.hist(y, alpha=0.5, bins=bins, density=True, label=y_name)
    plt.xlabel('Entropy', labelpad=10)
    plt.ylabel('Frequency', labelpad=10)
    plt.legend(frameon=False, loc='upper right', fontsize=16)
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if saveplots:
        path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/"
        plt.savefig(path + filename)

    if showplots:
        plt.show()

def display_statistics(labels, hall, non_hall, first_hall, subseq_hall):
    print("")
    print("Average Entropies")
    print("Hallucinated: %.4f" % np.mean(hall))
    print("Non-Hallucinated: %.4f" % np.mean(non_hall))
    print("Initial Hallucinated: %.4f" % np.mean(first_hall))
    print("Subsequent Hallucinated: %.4f" % np.mean(subseq_hall))

    print("")
    print("Non-Hallucinated vs Hallucinated, Initial Hallucinated, Subsequent Hallucinated")

    print("")
    print("T-test")
    t_test(hall, non_hall)
    t_test(first_hall, non_hall)
    t_test(subseq_hall, non_hall)

    print("")
    print("KS-test")
    ks_test(hall, non_hall)
    ks_test(first_hall, non_hall)
    ks_test(subseq_hall, non_hall)

    print("")
    print("Initial vs Subsequent Hallucinated")

    print("")
    print("T-test")
    t_test(subseq_hall, first_hall)

    print("")
    print("KS-test")
    ks_test(subseq_hall, first_hall)

def handle_plots(hall, non_hall, first_hall, subseq_hall, showplots, saveplots, plot_prefix):
    distrib(hall, non_hall, saveplots, showplots, filename=plot_prefix + "hallucinated_entropy_distribution")
    distrib(first_hall, non_hall, saveplots, showplots, filename=plot_prefix + "first_hallucination_entropy_distribution", x_name="First Hallucinated")
    distrib(subseq_hall, non_hall, saveplots, showplots, filename=plot_prefix + "subseq_vs_non_hallucination_entropy_distribution", x_name="Subsequent Hallucinated")
    distrib(first_hall, subseq_hall, saveplots, showplots, filename=plot_prefix + "first_vs_subseq_hallucination_entropy_distribution", x_name="First Hallucinated", y_name="Subsequent Hallucinated")

def remove_target(mylist, target):
    return [x for x in mylist if not x == 5.3129]

def display_results(labels, entropies, showplots=True, saveplots=False, plot_prefix="", drop_first=False):
    # Sanity check
    assert len(labels) == len(entropies)
    assert all(len(i) == len(j) for i, j in zip(labels, entropies))

    hall, non_hall = get_hall(labels, entropies)
    first_hall = get_first_hall(labels, entropies)
    subseq_hall = get_subseq_hall(labels, entropies)

    if (drop_first):
        # Remove first token from all sentences
        hall = remove_target(hall, 5.3129)
        non_hall = remove_target(non_hall, 5.3129)
        first_hall = remove_target(first_hall, 5.3129)

        # Sanity checks
        assert 5.3129 not in hall
        assert 5.3129 not in non_hall
        assert 5.3129 not in first_hall

    display_statistics(labels, hall, non_hall, first_hall, subseq_hall)
    handle_plots(hall, non_hall, first_hall, subseq_hall, showplots, saveplots, plot_prefix)


if __name__ == '__main__':
    base_path = "data/xsum-hallucination/"
    label_filename = base_path + "test.label"
    labels = read_lines(label_filename, int)

    entropy_filename = base_path + "test.entropy"
    entropies = read_lines(entropy_filename, float)
    display_results(labels, entropies, showplots=False, saveplots=False)

    entropy_filename = base_path + "test.entropy.lang_full"
    entropies = read_lines(entropy_filename, float)
    display_results(labels, entropies, showplots=False, saveplots=False, plot_prefix="lm_")

    display_results(labels, entropies, showplots=False, saveplots=True, plot_prefix="lm_filter_", drop_first=True)
