from tqdm import tqdm
import torch

def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    max_samples=100_000,
    grad_clip=1.0,
):
    model.train()
    total_samples = 0
    epoch = 0

    pbar = tqdm(total=max_samples, desc="Training samples")

    while total_samples < max_samples:
        epoch += 1

        for source_seq, target_input, target_output in train_loader:
            if total_samples >= max_samples:
                break

            batch_size = source_seq.size(0)

            source_seq = source_seq.to(device)
            target_input = target_input.to(device)
            target_output = target_output.to(device)

            optimizer.zero_grad()
            logits = model(source_seq, target_input)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_output.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_samples += batch_size
            pbar.update(batch_size)

    pbar.close()
