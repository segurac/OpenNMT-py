from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch
import subprocess

import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
if opt.batch_size != 1:
    print("WARNING: -batch_size isn't supported currently, "
          "we set it to 1 for now!")
    opt.batch_size = 1


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


def main():

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    print(opt)

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    print('*'*10 + ' Chatbot ' + '*'*10)
    # os.system("stty -echo")
    while True:
        myline = input('')
        auxfile = open('aux.txt', 'w')
        auxfile.write("%s\n" % myline)
        auxfile.close()

        if myline == 'exit':
            break

        p = subprocess.Popen('cat aux.txt | tools/tokenizer.perl -l en | tools/scripts/lowercase.perl > aux.tok',
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        p.wait()

        data = onmt.IO.ONMTDataset('aux.tok', None, translator.fields, None)

        test_data = onmt.IO.OrderedIterator(
            dataset=data, device=opt.gpu,
            batch_size=opt.batch_size, train=False, sort=False,
            shuffle=False)

        # length = len(list(test_data))
        for batch in test_data:
            pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
                = translator.translate(batch, data)
            pred_score_total += sum(score[0] for score in pred_scores)
            pred_words_total += sum(len(x[0]) for x in pred_batch)

            os.write(1, bytes('PRED: %s\n' % pred_batch[0][0], 'UTF-8'))

    # os.system("stty echo")

if __name__ == "__main__":
    main()
