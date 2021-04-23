import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from dialogue.io.DialogueDataset import DialogueDatasetIterator
from dialogue.toolbox.vocab import BOS_WORD, EOS_WORD


def model_infer(model_path, inp_path, outp_path):
    model_file = torch.load(model_path)
    train_opt = model_file["train_opt"]
    vocabs = model_file["vocabs"]

    meta_opt = train_opt.meta
    test_iter = DialogueDatasetIterator(
        file_name=inp_path, vocabs=vocabs,
        epochs=meta_opt.epochs, batch_size=1,
        is_train=False, n_workers=meta_opt.n_workers,
        use_cuda=meta_opt.use_cuda, opt=meta_opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_opt.gpu)

    model = model_file["model"]
    model.eval()
    model.flatten_parameters()

    pred_list = []
    total_loss = 0
    total_word_num = 0
    for batch in tqdm(test_iter):
        result_dict = model.run_batch(batch)
        total_loss += result_dict["loss"].item()
        total_word_num += result_dict["num_words"]
        preds_idx = model.predict_batch(
            batch, max_len=20, beam_size=5, eos_val=vocabs["word"].to_idx(EOS_WORD))
        preds = [[vocabs["word"].to_word(t) for t in item] for item in preds_idx]
        pred_list.extend(preds)

    per_word_loss = total_loss / total_word_num
    s1 = "Valid, Loss: {:.2f}, PPL: {:.2f}".format(np.log(model_file["score"]), model_file["score"])
    s2 = "Test, Loss: {:.2f}, PPL: {:.2f}".format(per_word_loss, np.exp(per_word_loss))
    print(s1)
    print(s2)
    with open(outp_path, "w") as f:
        for pred in pred_list:
            if BOS_WORD in pred:
                pred.remove(BOS_WORD)
            if EOS_WORD in pred:
                pred.remove(EOS_WORD)
            f.write(" ".join(pred) + "\n")
    with open(outp_path + ".ppl.txt", "w") as f:
        f.write(s1 + "\n")
        f.write(s2 + "\n")


if __name__ == "__main__":
    model_infer(sys.argv[1], sys.argv[2], sys.argv[3])