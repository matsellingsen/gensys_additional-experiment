import matplotlib.pyplot as plt

# EXPERIMENT 3

def exp3_plot_token_accuracy(results):
    ks = sorted(k for k in results.keys() if k not in [0, 99])
    token_means = [results[k]["tok"]["mean"] * 100 for k in ks]

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(ks)), token_means)
    plt.xticks(range(len(ks)), ks)
    plt.xlabel("Number of Composed Commands Used for Training")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.title("Token-Level Accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def exp3_plot_sequence_accuracy(results):
    ks = sorted(k for k in results.keys() if k not in [0, 99])
    sequence_means = [results[k]["seq"]["mean"] * 100 for k in ks]

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(ks)), sequence_means)
    plt.xticks(range(len(ks)), ks)
    plt.xlabel("Number of Composed Commands Used for Training")
    plt.ylabel("Accuracy on New Commands (%)")
    plt.title("Sequence-Level Accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def exp3_part1_plot_token_accuracy(results):
    labels = list(results.keys())
    token_means = [results[label]["avg_tok_acc"] * 100 for label in labels]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, token_means)
    plt.ylabel("Token Accuracy (%)")
    plt.title("Experiment 3 – Part 1: Token-Level Accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def exp3_part1_plot_sequence_accuracy(results):
    labels = list(results.keys())
    seq_means = [results[label]["avg_seq_acc"] * 100 for label in labels]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, seq_means)
    plt.ylabel("Sequence Accuracy (%)")
    plt.title("Experiment 3 – Part 1: Sequence-Level Accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# EXPERIMENT 1

def exp1_plot_sequence_accuracy(results):
    """
    Plot sequence-level accuracy vs percentage of commands used for training.
    """
    percentages = sorted(results.keys())
    mean_accuracies = [results[p]["mean"] * 100 for p in percentages]

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(percentages)), mean_accuracies)
    plt.xticks(range(len(percentages)), percentages)
    plt.xlabel("Percentage of Commands Used for Training")
    plt.ylabel("Sequence Accuracy (%)")
    plt.title("Experiment 1: Sequence-Level Accuracy vs Training Coverage")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# EXPERIMENT 2

def _extract_exp2_means(final_results, split_type, oracle, metric):
    """
    metric: 'seq' or 'tok'
    """
    oracle_key = "with_oracle" if oracle else "without_oracle"

    data = final_results[split_type][oracle_key]

    lengths = sorted(data.keys())
    means = [data[L][metric]["mean"] * 100 for L in lengths]

    return lengths, means

def exp2_plot_sequence_accuracy(final_results, split_type, oracle=False):
    lengths, means = _extract_exp2_means(
        final_results,
        split_type=split_type,
        oracle=oracle,
        metric="seq",
    )

    oracle_str = "with oracle" if oracle else "without oracle"
    title = f"Sequence Accuracy by {split_type.replace('_', ' ').title()} ({oracle_str})"

    plt.figure(figsize=(7, 5))
    plt.bar(lengths, means)
    plt.xlabel("Length")
    plt.ylabel("Sequence Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def exp2_plot_token_accuracy(final_results, split_type, oracle=False):
    lengths, means = _extract_exp2_means(
        final_results,
        split_type=split_type,
        oracle=oracle,
        metric="tok",
    )

    oracle_str = "with oracle" if oracle else "without oracle"
    title = f"Token Accuracy by {split_type.replace('_', ' ').title()} ({oracle_str})"

    plt.figure(figsize=(7, 5))
    plt.bar(lengths, means)
    plt.xlabel("Length")
    plt.ylabel("Token Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()





