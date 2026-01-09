import os
import requests
#import argparse

def download_github_folder(repo_owner, repo_name, folder_path, output_dir):
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
            print(f"Downloaded file: {out_path}")

        elif item["type"] == "dir":
            # Recursively download subfolders
            download_github_folder(
                repo_owner,
                repo_name,
                item["path"],
                os.path.join(output_dir, item["name"])
            )

# Example usage:
download_github_folder(
    repo_owner="brendenlake",
    repo_name="SCAN",
    folder_path="length_split",
    output_dir="data/length_split"
)

def main(repo_owner="brendenlake", repo_name="SCAN", folder_path="length_split", output_dir="length_split"):
    """parser = argparse.ArgumentParser(description="Download a folder from a GitHub repository.")
    parser.add_argument("repo_owner", type=str, help="Owner of the GitHub repository.")
    parser.add_argument("repo_name", type=str, help="Name of the GitHub repository.")
    parser.add_argument("folder_path", type=str, help="Path to the folder in the repository.")
    parser.add_argument("output_dir", type=str, help="Local directory to save the downloaded files.")

    args = parser.parse_args()"""

    output_dir = os.path.join("data", output_dir)
    
    download_github_folder(
        repo_owner=repo_owner,
        repo_name=repo_name,
        folder_path=folder_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()