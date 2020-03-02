'''
API for generating adverserial examples of word flips
and using them to deteriorate an lstm classifier
'''

import os
import re
import sys
import subprocess

ONE_WORD_DIR = 'save/one_word'
TWO_WORD_DIR = 'save/two_word'

def generate_adv_examples(model_type='one_word', device='cuda'):
    subprocess.call([
        "python",
        "use_pretrained_gene_testset.py",
        "-nonstatic",
        "-word2vec",
        model_type,
        "THEANO_FLAGS=mode=FAST_RUN,device=" + device + ",floatX=float32"
    ])
    clean_raw_examples(model_type)

def clean_raw_examples(model_type='one_word'):
    if model_type == 'one_word':
        dir = ONE_WORD_DIR
    else:
        dir = TWO_WORD_DIR
    read_file = open(dir+'/sst2_0.4_two_examples_raw.txt', 'r')
    write_file = open(dir+'/sst2_0.4_two_examples.txt', 'w+')

    for line in read_file.readlines():
        clean_line = re.sub(r'(?is) chr201', '', line.strip())
        write_file.write(clean_line+'\n')

    read_file.close()
    write_file.close()

def attack_lstm_classifier(model_type='one_word', orig='original', device='cuda'):
    subprocess.call([
        "python",
        "lstm/use_pretrained_model.py",
        model_type,
        orig,
        "THEANO_FLAGS=mode=FAST_RUN,device=" + device + ",floatX=float32"
    ])

if __name__ == '__main__':
    while True:
        print '====================================================='
        print '                       HOTFLIP                       '
        print '          ADVERSERIAL TEXT NOISE GENERATOR           '
        print '====================================================='
        print '1. Generate examples from one word model'
        print '2. Generate examples from two word model'
        print '3. Test one word trained attack'
        print '4. Test two word trained attack'
        print '5. Benchmark lstm classifier on original testset'
        print '6. QUIT'
        choice = raw_input('Enter chocie [1-6] : ')
        choice = int(choice)

        if choice == 1:
            generate_adv_examples(model_type='one_word')
        elif choice == 2:
            generate_adv_examples(model_type='two_word')
        elif choice == 3:
            attack_lstm_classifier(model_type='one_word', orig='adverserial')
        elif choice == 4:
            attack_lstm_classifier(model_type='two_word', orig='adverserial')
        elif choice == 5:
            attack_lstm_classifier()
        elif choice == 6:
            sys.exit()
        else:
            sys.exit()
