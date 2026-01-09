import os
import re
import requests

# SCAN special token indices
PAD = 0
BOS = 1
EOS = 2

SPECIAL_TOKENS = {
    "<PAD>": PAD,
    "<BOS>": BOS,
    "<EOS>": EOS,
}

########################################
# Download utilities
########################################

def download_github_folder(repo_owner, repo_name, folder_path, output_dir):
    """
    Download a folder from a public GitHub repo using the GitHub API.
    Saves all files (recursively) into output_dir.
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()

    os.makedirs(output_dir, exist_ok=True)

    for item in items:
        if item["type"] == "file":
            file_url = item["download_url"]
            file_data = requests.get(file_url).content
            out_path = os.path.join(output_dir, item["name"])

            with open(out_path, "wb") as f:
                f.write(file_data)

        elif item["type"] == "dir":
            download_github_folder(
                repo_owner,
                repo_name,
                item["path"],
                os.path.join(output_dir, item["name"]),
            )


def download_scan_simple_split(base_dir="data"):
    """
    Download SCAN simple_split into base_dir/simple_split
    """
    output_dir = os.path.join(base_dir, "simple_split")

    train_path = os.path.join(output_dir, "tasks_train_simple.txt")
    test_path  = os.path.join(output_dir, "tasks_test_simple.txt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return output_dir

    download_github_folder(
        repo_owner="brendenlake",
        repo_name="SCAN",
        folder_path="simple_split",
        output_dir=output_dir,
    )
    return output_dir

def download_scan_add_prim_split(base_dir="data"):
    """
    Download SCAN simple_split into base_dir/simple_split
    """
    output_dir = os.path.join(base_dir, "add_prim_split")

    train_path = os.path.join(output_dir, "tasks_train_addprim_jump.txt")
    test_path  = os.path.join(output_dir, "tasks_test_addprim_jump.txt")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return output_dir

    download_github_folder(
        repo_owner="brendenlake",
        repo_name="SCAN",
        folder_path="add_prim_split",
        output_dir=output_dir,
    )
    return output_dir


########################################
# Parsing utilities
########################################

def load_scan_split(path):
    """
    Load a SCAN tasks file.
    Returns:
      inputs:  List[List[str]]
      outputs: List[List[str]]
    """
    inputs, outputs = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            inp = re.findall(r"IN: (.*) OUT:", line)[0]
            out = line.split("OUT: ")[1]

            inputs.append(inp.split())
            outputs.append(out.split())

    return inputs, outputs


########################################
# Vocabulary utilities
########################################

def build_vocab(seqs):
    """
    Build token -> index mapping from training sequences only.
    """
    vocab = dict(SPECIAL_TOKENS)
    idx = max(vocab.values()) + 1

    for seq in seqs:
        for tok in seq:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1

    return vocab


def encode(seq, vocab):
    return [vocab[t] for t in seq]


def make_decoder_inputs(seq_ids):
    """
    Given target token IDs:
      [y1, y2, ..., yN]

    Return:
      tgt_in  = [BOS, y1, ..., yN]
      tgt_out = [y1, ..., yN, EOS]
    """
    return [BOS] + seq_ids, seq_ids + [EOS]
