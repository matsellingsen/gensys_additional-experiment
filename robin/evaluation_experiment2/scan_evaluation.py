import torch
from data_utils.scan_helpers import PAD, BOS, EOS


@torch.no_grad()
def greedy_decode(model, source_sequences, use_oracle=False, target_output_lengths=None, max_length=100):
    model.eval()
    batch_size = source_sequences.size(0)

    decoded_sequences = torch.full(
        (batch_size, 1),
        BOS,
        dtype=torch.long,
        device=source_sequences.device,
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=source_sequences.device)

    for step in range(max_length):
        logits = model(source_sequences, decoded_sequences)
        next_tokens = logits[:, -1].argmax(dim=-1, keepdim=True)

        
        # Apply oracle constraints if specified
        if use_oracle:
            assert target_output_lengths is not None

            for i in range(batch_size):
                # Too early → forbid EOS
                if step + 1 < target_output_lengths[i] and next_tokens[i] == EOS:
                    # choose best non-EOS token instead
                    logits_i = logits[i, -1]
                    logits_i[EOS] = float("-inf")
                    next_tokens[i] = logits_i.argmax()

                # At or after target length → force EOS
                if step + 1 >= target_output_lengths[i]:
                    next_tokens[i] = EOS

        finished |= (next_tokens.squeeze(1) == EOS)
        next_tokens[finished] = EOS

        decoded_sequences = torch.cat([decoded_sequences, next_tokens], dim=1)

        if finished.all():
            break

    return decoded_sequences


def sequence_accuracy(predicted_sequences, target_sequences):
    correct = 0

    for predicted, target in zip(predicted_sequences, target_sequences):
        predicted = predicted.tolist()
        target = target.tolist()

        # Remove BOS if present
        if len(predicted) > 0 and predicted[0] == BOS:
            predicted = predicted[1:]

        if EOS in predicted:
            predicted = predicted[: predicted.index(EOS) + 1]
        if EOS in target:
            target = target[: target.index(EOS) + 1]

        if predicted == target:
            correct += 1

    return correct / len(target_sequences)


@torch.no_grad()
def evaluate_sequence_accuracy(model, dataloader, device, use_oracle=False):
    model.eval()
    total_correct = 0
    total_samples = 0

    for source_sequences, _, target_output in dataloader:
        source_sequences = source_sequences.to(device)
        target_output = target_output.to(device)
        """print("target_output:", target_output)
        for tensor in target_output:
            print("  ", len(tensor.tolist()))
        target_output_lengths = (target_output != PAD).sum(dim=1)
        print("Target output lengths:", target_output_lengths.tolist())
        # stop the whole code here, its just for testing.

        raise SystemExit("Stopping execution - TESTING")"""
        target_output_lengths = (target_output != PAD).sum(dim=1)
        predictions = greedy_decode(model, source_sequences, use_oracle=use_oracle, target_output_lengths = target_output_lengths, max_length=100)

        batch_accuracy = sequence_accuracy(predictions, target_output)
        total_correct += batch_accuracy * source_sequences.size(0)
        total_samples += source_sequences.size(0)

    return total_correct / total_samples


def tokens_accuracy(logits, targets): # Forced-teacher app
    """
    logits:  [batch, length, vocab]
    targets: [batch, length]
    """
    
    predictions = logits.argmax(dim=-1)
    mask = targets != PAD

    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def evaluate_token_level(model, dataloader, criterion, device, use_oracle=False, max_length=100):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0

    for source_sequences, target_input, target_output in dataloader:
        source_sequences = source_sequences.to(device)
        target_input = target_input.to(device)
        target_output = target_output.to(device)

        if use_oracle:
            # Teacher-forced, oracle-length
            logits = model(source_sequences, target_input)

            if isinstance(logits, tuple):
                logits = logits[0]  # take only the actual logits tensor
            
            #print("logits shape:", logits.shapes)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_output.reshape(-1),
            )
            accuracy = tokens_accuracy(logits, target_output)

        else:
            # Free decoding (autoregressive generation without oracle)
            decoded_sequences = greedy_decode(
                model,
                source_sequences,
                use_oracle=False,
                max_length=max_length,
            )

            # Remove BOS and align with target_output for token-level comparison
            max_len = target_output.size(1)
            decoded_truncated = decoded_sequences[:, 1:]  # skip BOS

            # Pad or truncate to match target length
            if decoded_truncated.size(1) < max_len:
                pad_size = max_len - decoded_truncated.size(1)
                padding = torch.full(
                    (decoded_truncated.size(0), pad_size),
                    PAD,
                    dtype=decoded_truncated.dtype,
                    device=decoded_truncated.device
                )
                decoded_truncated = torch.cat([decoded_truncated, padding], dim=1)
            else:
                decoded_truncated = decoded_truncated[:, :max_len]

            # Compare predictions to targets at token level
            # Note: We do NOT mask after prediction EOS - if the model stops early,
            # all subsequent positions (PAD in prediction vs. real tokens in target) 
            # should count as errors
            mask = target_output != PAD
            correct = (decoded_truncated == target_output) & mask
            accuracy = correct.sum().item() / max(1, mask.sum().item())

            # Loss is not meaningful for autoregressive generation
            # (predictions are not valid decoder inputs)
            

        
        total_accuracy += accuracy
        steps += 1

    return "placeholder" , total_accuracy / steps

def ordered_sequence_token_accuracy(predicted_sequences, target_sequences):
    """
    Partial-credit, position-wise token accuracy on decoded sequences.
    """
    total_correct = 0
    total_tokens = 0

    for predicted, target in zip(predicted_sequences, target_sequences):
        predicted = predicted.tolist()
        target = target.tolist()

        # Remove BOS if present
        if len(predicted) > 0 and predicted[0] == BOS:
            predicted = predicted[1:]

        # Truncate at EOS
        if EOS in predicted:
            predicted = predicted[: predicted.index(EOS)]
        if EOS in target:
            target = target[: target.index(EOS)]

        # Compare position-wise
        L = min(len(predicted), len(target))
        for i in range(L):
            if predicted[i] == target[i]:
                total_correct += 1
            total_tokens += 1

        # If prediction is shorter than target, remaining tokens count as incorrect
        total_tokens += max(0, len(target) - L)

    return total_correct / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def evaluate_ordered_token_accuracy(model, dataloader, device):
    model.eval()
    total_acc = 0.0
    steps = 0

    for source_sequences, _, target_output in dataloader:
        source_sequences = source_sequences.to(device)
        target_output = target_output.to(device)

        predictions = greedy_decode(model, source_sequences)
        acc = ordered_sequence_token_accuracy(predictions, target_output)

        total_acc += acc
        steps += 1

    return total_acc / steps

