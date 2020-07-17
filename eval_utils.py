""" Define necessary functions for evaluation.
"""
import torch
import editdistance


def get_phn_mapping_table():
    """
    Build the phoneme mapping table.
    Sequences are mapped from 61 to 39 phonemes during evaluation.
    This mapping is a standard recipe taken from the Kaldi TIMIT s5 recipe:
    https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/phones.60-48-39.map
    """
    table = {}
    with open('phones.60-48-39.map') as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]
    for l in lines:
        if len(l) == 3:
            table[l[0]] = l[2]
    return table


def mapping(s_in, table):
    """
    Mapping a sequence from 61 to 39 phonemes.

    Args:
        s_in (string): Original sentence.

    Returns:
        s_out (list(string)): Decoded sentence (words).
    """
    s_out = []
    for w in s_in.split():
        if w in table:
            s_out.append(table[w])
    return s_out


def get_error(dataloader, model):
    """
    Calculate error rate on a specific dataset.
    """
    tokenizer = torch.load('tokenizer.pth')
    table = get_phn_mapping_table()
    n_tokens = 0
    total_error = 0
    with torch.no_grad():
        for i, (xs, xlens, ys) in enumerate(dataloader):
            preds_batch, _ = model(xs.cuda(), xlens)   # [batch_size, 100]
            for j in range(preds_batch.shape[0]):
                preds = tokenizer.decode(preds_batch[j])
                gt = tokenizer.decode(ys[j])

                # Sequences are mapped from 61 to 39 phonemes during evaluation.
                preds = mapping(preds, table)
                gt = mapping(gt, table)

                total_error += editdistance.eval(gt, preds)
                n_tokens += len(gt)
            print ("Calculating error rate ... (#batch: %d/%d)" % (i+1, len(dataloader)), end='\r')
    print ()
    error = total_error / n_tokens
    return error

