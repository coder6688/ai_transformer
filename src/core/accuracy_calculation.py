import torch

def calculate_accuracy(preds, labels, pad_token_id):
    # preds: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    
    # Get predicted token indices
    _, predicted = torch.max(preds, dim=-1)
    
    # Create mask for non-pad tokens
    non_pad_mask = labels != pad_token_id
    
    # Calculate correct predictions (only where mask is True)
    correct = (predicted == labels) & non_pad_mask
    
    # Compute accuracy
    accuracy = correct.sum().float() / non_pad_mask.sum().float()
    return accuracy.item()
