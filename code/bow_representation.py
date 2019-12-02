"""
===================
bow_representation.py
===================
Create bag-of-words (bow) bigram/unigram representation of logic expressions
"""

import sys
from create_expressions import LogicTreeTrainer, LogicTree, LogicNode, TrueNode
from create_expressions import FalseNode, PNode, QNode, RNode, AndNode, OrNode
from create_expressions import ImplicationNode, NotNode
import pickle as pkl
import string
import random
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
import scipy.io
import copy



class Bag_of_words():

    def __init__(self, source=None, out=None):

        all_symbols = ['p', 'q', 'r']
        all_symbols.append('T')
        all_symbols.append('F')
        all_symbols.extend(['∧', '∨', '→', '~'])
        all_symbols.extend(['(', ')'])
        all_symbols.extend(['STA', 'EOS'])


        self.symbols = {}
        for symbol in all_symbols:
            self.symbols[symbol] = len(self.symbols)
        self.bigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                self.bigrams[(symbol1, symbol2)] = len(self.bigrams)
        self.trigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                for symbol3 in all_symbols:
                    self.trigrams[(symbol1, symbol2, symbol3)] = len(self.trigrams)

            self.trigrams[('STA', symbol1, None)] = len(self.trigrams)
            self.trigrams[(None, symbol1, 'EOS')] = len(self.trigrams)


        if source == None:
            self.trainer = pkl.load(open('../data/trainer.pkl', 'rb'))
        else:
            self.source = source
            source_path = '../data/logic_proof_datasets/' + source + '.pkl'
            self.trainer = pkl.load(open(source_path, 'rb'))

        if out == None:
            self.dump_loc = '../data/'
        else:
            self.dump_loc = out

        self.dataset = self.trainer.trees
        print(len(self.dataset))
        self.check_valid_dataset()

    def check_valid_dataset(self):
        for tree_id, trees in self.dataset.items():
            for symbol in trees[0].parse_tree():
                if symbol not in self.symbols:
                    print(symbol)
                    assert symbol in self.symbols
            for treetup in trees[1]:
                for symbol in treetup[0].parse_tree():
                    if symbol not in self.symbols:
                        print(symbol)
                        assert symbol in self.symbols

    def bow_unigram(self, expr):
        unigrams_count = defaultdict(int)
        for i in range(0, len(expr)):
            unigrams_count[expr[i]] += 1

        unigram_index_count = [0] * len(self.symbols)
        for expr in unigrams_count.keys():
            unigram_index_count[self.symbols[expr]] = unigrams_count[expr]
        return unigram_index_count

    def bow_bigram(self, expr):
        bigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 1):
            bigrams_count[(expr[i], expr[i + 1])] += 1
        bigrams_count[('STA', expr[0])] += 1
        bigrams_count[(expr[len(expr) - 1], 'EOS')] += 1

        bigram_index_count = [0] * len(self.bigrams)
        for (expr1, expr2) in bigrams_count.keys():
            bigram_index_count[self.bigrams[(expr1, expr2)]] = bigrams_count[(expr1, expr2)]
        return bigram_index_count

    def bow_trigram(self, expr):
        trigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 2):
            trigrams_count[(expr[i], expr[i + 1], expr[i + 2])] += 1
        try:
            trigrams_count[('STA', expr[0], expr[1])] += 1
        except IndexError:
            trigrams_count[('STA', expr[0], None)] += 1
        try:
            trigrams_count[(expr[len(expr) - 2], expr[len(expr) - 1], 'EOS')] += 1
        except IndexError:
            trigrams_count[(None, expr[len(expr) - 1], 'EOS')] += 1

        trigram_index_count = [0] * len(self.trigrams)
        for (expr1, expr2, expr3) in trigrams_count.keys():
            trigram_index_count[self.trigrams[(expr1, expr2, expr3)]] = trigrams_count[(expr1, expr2, expr3)]
        return trigram_index_count

    def bow_representation_expr(self, expr, trigrams=False):
        if not trigrams:
            return self.bow_unigram(expr) + self.bow_bigram(expr)
        else:
            return self.bow_unigram(expr) + self.bow_bigram(expr) + self.bow_trigram(expr)

    def bow_representation_bigram_dataset(self, dump=None):
        print("creating", self.source, "bigram dataset")
        bow_dataset = {}
        for tree_id, trees in self.dataset.items():
            tree_bow = self.bow_representation_expr(trees[0].parse_tree())
            tree_sequence = trees[1]
            new_tree_sequence = []
            for treetup in tree_sequence:
                new_tree_sequence.append((treetup[0].parse_tree(), treetup[1], \
                    self.bow_representation_expr(treetup[0].parse_tree())))
            bow_dataset[tree_id] = ((trees[0].parse_tree(), tree_bow), new_tree_sequence)
        print("dumping")
        pkl.dump(bow_dataset, open(self.dump_loc + '_bigram_dataset.pkl', 'wb'))

    def bow_representation_trigram_dataset(self, percentage):
        bow_dataset = {}
        for class_id, equiv_class in self.dataset.items():
            if(random.randint(1,100) <= percentage):
                bow_equiv_class = []
                for expr in equiv_class:
                    bow_equiv_class.append((expr[0], expr[1], expr[2], self.bow_representation_expr(expr[2], trigrams=True)))
                bow_dataset[class_id] = bow_equiv_class
                print(len(bow_dataset))
        print("dumping")
        pkl.dump(bow_dataset, open('../data/trigram_dataset.pkl', 'wb'))

    def bow_collisions(self, percentage=100, trigrams=False):
        if not trigrams:
            try:
                bows = pkl.load(open('../data/bigram_dataset.pkl', 'rb'))
            except FileNotFoundError:
                print("dataset not found, creating a new one")
                self.bow_representation_bigram_dataset(percentage)
                bows = pkl.load(open('../data/bigram_dataset.pkl', 'rb'))
        else:
            try:
                bows = pkl.load(open('../data/trigram_dataset.pkl', 'rb'))
            except FileNotFoundError:
                self.bow_representation_trigram_dataset(percentage)
                bows = pkl.load(open('../data/trigram_dataset.pkl', 'rb'))

        stats_file = open('../data/dataset_stats/stats.txt', 'w')

        bows_vals = list(bows.values())
        op_depths = [len(tup[1]) for tup in bows_vals]
        op_depths_dict = {depth:[] for depth in range(min(op_depths), max(op_depths)+1)}

        for idx in range(len(bows_vals)):
            op_depths_dict[op_depths[idx]].append(bows_vals[idx][1])

        for op_depth, bigrams in op_depths_dict.items():

            duplicates = []
            freqs = []

            for idx1 in range(len(bigrams)-1):
                bigram = bigrams[idx1]
                if bigram not in duplicates:
                    freq_count = 1
                    found = False
                    for idx2 in range(idx1+1, len(bigrams)):
                        if bigram == bigrams[idx2]:
                            found = True
                            freq_count += 1
                    if found:
                        duplicates.append(bigram)
                        freqs.append(freq_count)

            sum_duplicates = sum([freq for freq in freqs])
            write_string = "operation depth " + str(op_depth) + "%duplicates: "\
                            + str(100*(sum_duplicates/len(bigrams)))
            stats_file.write(write_string)
            stats_file.write("\n")

        print("closing stats file")
        stats_file.close()







class BOW_exprs():

    def __init__(self, source=None, out=None):


        all_symbols = ['p', 'q', 'r']
        all_symbols.append('T')
        all_symbols.append('F')
        all_symbols.extend(['∧', '∨', '→', '~'])
        all_symbols.extend(['(', ')'])
        all_symbols.extend(['STA', 'EOS'])



        self.symbols = {}
        for symbol in all_symbols:
            self.symbols[symbol] = len(self.symbols)
        self.bigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                self.bigrams[(symbol1, symbol2)] = len(self.bigrams)
        self.trigrams = {}
        for symbol1 in all_symbols:
            for symbol2 in all_symbols:
                for symbol3 in all_symbols:
                    self.trigrams[(symbol1, symbol2, symbol3)] = len(self.trigrams)

            self.trigrams[('STA', symbol1, None)] = len(self.trigrams)
            self.trigrams[(None, symbol1, 'EOS')] = len(self.trigrams)


    def bow_unigram(self, expr):
        unigrams_count = defaultdict(int)
        for i in range(0, len(expr)):
            unigrams_count[expr[i]] += 1

        unigram_index_count = [0] * len(self.symbols)
        for expr in unigrams_count.keys():
            unigram_index_count[self.symbols[expr]] = unigrams_count[expr]
        return unigram_index_count


    def bow_bigram(self, expr):
        bigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 1):
            bigrams_count[(expr[i], expr[i + 1])] += 1
        bigrams_count[('STA', expr[0])] += 1
        bigrams_count[(expr[len(expr) - 1], 'EOS')] += 1

        bigram_index_count = [0] * len(self.bigrams)
        for (expr1, expr2) in bigrams_count.keys():
            bigram_index_count[self.bigrams[(expr1, expr2)]] = bigrams_count[(expr1, expr2)]
        return bigram_index_count

    def bow_trigram(self, expr):
        trigrams_count = defaultdict(int)
        for i in range(0, len(expr) - 2):
            trigrams_count[(expr[i], expr[i + 1], expr[i + 2])] += 1
        try:
            trigrams_count[('STA', expr[0], expr[1])] += 1
        except IndexError:
            trigrams_count[('STA', expr[0], None)] += 1
        try:
            trigrams_count[(expr[len(expr) - 2], expr[len(expr) - 1], 'EOS')] += 1
        except IndexError:
            trigrams_count[(None, expr[len(expr) - 1], 'EOS')] += 1

        trigram_index_count = [0] * len(self.trigrams)
        for (expr1, expr2, expr3) in trigrams_count.keys():
            trigram_index_count[self.trigrams[(expr1, expr2, expr3)]] = trigrams_count[(expr1, expr2, expr3)]
        return trigram_index_count

    def bow_representation_expr(self, expr, trigrams=False):
        if not trigrams:
            return self.bow_unigram(expr) + self.bow_bigram(expr)
        else:
            return self.bow_unigram(expr) + self.bow_bigram(expr) + self.bow_trigram(expr)












if __name__ == '__main__':

    # starting_exprs = ['T', 'F', 'p', 'q', 'r', '~p', '~q', '~r', 'p→q']
    # starting_exprs = ['p→q']
    starting_exprs = ['T']

    for expr in starting_exprs:

        dump_loc = '../data/bigram_datasets/' + expr

        bow = Bag_of_words(expr, dump_loc)
        bow.bow_representation_bigram_dataset()



































# comment
