import pickle
import csv
import time
import math
import pprint
import numpy as np
from scipy.stats import rankdata
import sys
from functools import reduce
from sklearn.externals import joblib

from wiki_dsmb.trie_element import TrieElement
from wiki_dsmb.token_trie import TokenTrie

class Wiki:
    #pages = {}

    def __init__(self, path, min_sense_prob=0.01, output=sys.stdout):
        self.data_path = path
        self.min_sense_prob = min_sense_prob
        self.output = output
        self.relatedness_cache = {}
        self.sense_cache = {}
        self.concepts_cache = {}
        self.use_cache = False

    def load(self):
        self._load_pages()
        self._load_links()
        self._load_occurrences()
        self._load_counts()
        self._load_token_trie()
        self._load_redirects()
        self._load_senses()
        self._load_concepts()
        self._load_disambiguation_model()
        self._load_indices()


    def print_senses(self, link):
        pprint.pprint({self.pages[k][0] : v for k,v in self.senses[link].items()})

    def link_probability(self, word):
        try:
            return self.occurrences_counts[word] / self.tokens_counts[word]
        except KeyError:
            return 0

    def semantic_relatedness(self, concept_id_1, concept_id_2):
        if(concept_id_1 == concept_id_2):
            return 1.0
        if(concept_id_1 > concept_id_2):
            concept_id_1, concept_id_2 = concept_id_2, concept_id_1
        if(self.use_cache):
            if((concept_id_1, concept_id_2) in self.relatedness_cache):
                return self.relatedness_cache[(concept_id_1, concept_id_2)]
        try:
            links_1 = self.links[concept_id_1]
            links_2 = self.links[concept_id_2]
        except KeyError:
            return self._set_relatedness_cache(concept_id_1, concept_id_2, 0.0)
        common_links_count = len(links_1.intersection(links_2))
        if(common_links_count == 0):
            return self._set_relatedness_cache(concept_id_1, concept_id_2, 0.0)
        value = 1 /(1 - math.log(common_links_count / (len(links_1) + len(links_2) - common_links_count)))
        return self._set_relatedness_cache(concept_id_1, concept_id_2, value)

    def _set_relatedness_cache(self, concept_id_1, concept_id_2, value):
        if(self.use_cache):
            self.relatedness_cache[(concept_id_1, concept_id_2)] = 0.0
        return value

    def semantic_relatedness_str(self, concept_str_1, concept_str_2):
        try:
            concept_id_1 = self.pages_reverse[concept_str_1]
            concept_id_2 = self.pages_reverse[concept_str_2]
            return self.semantic_relatedness(concept_id_1, concept_id_2)
        except KeyError:
            return 0


    def resolve_redirect(self, source_id):
        target_id = self.redirects[source_id]
        if(target_id == source_id):
            return source_id
        if(self.pages[target_id][1] == 2):
            return self.resolve_redirect(target_id)
        else:
            return target_id

    def print_concepts(self,concept_str):
        pprint.pprint(self.concepts[self.pages_reverse[concept_str]])

    def sense_probability(self, link, sense_id):
        if(self.use_cache and link in self.sense_cache):
            total_and_ranks = self.sense_cache[link]
        else:
            total = reduce(lambda acc, value: acc + value, self.senses[link].values(), 0.0)
            rank = 0
            ranks = {}
            last_value = 0
            for value in reversed(sorted(self.senses[link].values())):
                if(value != last_value):
                    ranks[value] = rank
                last_value = value
                rank += 1
            total_and_ranks = (total, ranks)
            if(self.use_cache):
                self.sense_cache[link] = total_and_ranks
        return (self.senses[link][sense_id] / total_and_ranks[0], total_and_ranks[1][self.senses[link][sense_id]])

    def sense_probability_str(self, link, sense_str):
        return self.sense_probability(link, self.pages_reverse[sense_str])

    def concept_probability(self, concept_id, link):
        if(self.use_cache and concept_id in self.concepts_cache):
            total_and_ranks = self.concepts_cache[concept_id]
        else:
            total = reduce(lambda acc, value: acc + value, self.concepts[concept_id].values(), 0.0)
            rank = 0
            ranks = {}
            last_value = 0
            for value in reversed(sorted(self.concepts[concept_id].values())):
                if(value != last_value):
                    ranks[value] = rank
                last_value = value
                rank += 1
            total_and_ranks = (total, ranks)
            if(self.use_cache):
                self.concepts_cache[concept_id] = total_and_ranks
        return (self.concepts[concept_id][link] / total_and_ranks[0], total_and_ranks[1][self.concepts[concept_id][link]])

    def concept_probability_str(self, concept_str, link):
        return self.concept_probability(self.pages_reverse[concept_str], link)

    def compute_weights_and_goodness(self, concept_block):
        weights = np.zeros((2, len(concept_block)), dtype=float)
        for concept_index in range(len(concept_block)):
            link, sense_id, _, _ = concept_block[concept_index]
            weights[0][concept_index] = self.link_probability(link)
            for other_concept_index in range(len(concept_block)):
                if(concept_index == other_concept_index):
                    continue
                _, other_sense_id, _, _ = concept_block[other_concept_index]
                rel = self.semantic_relatedness(sense_id, other_sense_id)
                #print("%s -- %s -- %.3f" % (pages[sense_id][0],  pages[other_sense_id][0], rel))
                weights[1][concept_index] += rel

        total_sem_rel = np.sum(weights[1])
        maxes = np.zeros(2, dtype=float)
        maxes[0] = np.max(weights[0])
        if(maxes[0] == 0):
            maxes[0] = 1
        maxes[1] = np.max(weights[1])
        if(maxes[1] == 0):
            maxes[1] = 1
        weights = np.transpose(weights)
        np.divide(weights, maxes, out=weights)
        weights = np.transpose(weights)
        return (np.add(weights[0], weights[1]), total_sem_rel)

    def compute_known_features(self, existing_links, weights, goodness):
        result = np.zeros((0, 9), dtype=float)
        for link_index in range(len(existing_links)):
            link, valid_sense_id, start_index, end_index = existing_links[link_index]
            if(link not in self.senses):
                continue
            if(len(self.senses[link]) == 1):
                continue
            l_probability = self.link_probability(link)
            local_block = np.zeros((0,9), dtype=float)
            # we only process links with alternative senses
            for sense_id in self.senses[link].keys():
                s_prob, s_rank = self.sense_probability(link, sense_id)
                if(s_prob < self.min_sense_prob):
                    continue
                c_prob, c_rank = self.concept_probability(sense_id, link)
                sem_rel_arr = np.zeros(len(existing_links), dtype=float)
                # semantic rel. computation
                for other_link_index in range(len(existing_links)):
                    if(link_index == other_link_index):
                        continue
                    _, other_sense_id, _, _ = existing_links[other_link_index]
                    sem_rel_arr[other_link_index] = self.semantic_relatedness(sense_id, other_sense_id)
                sem_rel = np.average(np.multiply(sem_rel_arr, weights))
                if(sense_id == valid_sense_id):
                    validity = 1
                else:
                    validity = 0
                local_block = np.append(local_block, [[validity, l_probability, goodness, s_prob, s_rank, c_prob, c_rank, sem_rel, 0]], axis=0)
            # some senses were skipped and there is only one example
            if(local_block.shape[0] <= 1):
                continue
            local_block[:,8] = rankdata(local_block[:,7], method='min')
            # flip the ranking and start with 0
            # it's important to flip, since otherwise the most connected senses won't have the same ranks
            local_block[:,8] = np.subtract(np.max(local_block[:,8]), local_block[:,8])
            result = np.append(result, local_block, axis=0)
        return result

    def compute_features(self, potential_links, weights, goodness, existing_links):
        features = np.zeros((0, 8), dtype=float)
        candidates = {}
        for link in potential_links:
            if(link not in self.senses):
                continue
            l_probability = self.link_probability(link)
            local_block = np.zeros((0,8), dtype=float)
            for sense_id in self.senses[link].keys():
                s_prob, s_rank = self.sense_probability(link, sense_id)
                if(s_prob < self.min_sense_prob):
                    continue
                c_prob, c_rank = self.concept_probability(sense_id, link)
                sem_rel_arr = np.zeros(len(existing_links), dtype=float)
                # when moving keep in mind that the list cannot be empty
                if(link not in candidates):
                    candidates[link]  = []
                candidates[link].append(sense_id)
                # semantic rel. computation
                for other_link_index in range(len(existing_links)):
                    _, other_sense_id, _, _ = existing_links[other_link_index]
                    sem_rel_arr[other_link_index] = self.semantic_relatedness(sense_id, other_sense_id) * weights[other_link_index]
                sem_rel = np.average(sem_rel_arr)
                local_block = np.append(local_block, [[l_probability, goodness, s_prob, s_rank, c_prob, c_rank, sem_rel, 0]], axis=0)
            if(len(local_block) > 0):
                local_block[:,7] = rankdata(local_block[:,6], method='min')
                local_block[:,7] = np.subtract(np.max(local_block[:,7]), local_block[:,7])
                features = np.append(features, local_block, axis=0)
        return (features, candidates)

    def _load_pages(self):
        if(hasattr(self, "pages")):
            return
        self._print("Loading pages")
        start_time = time.time()
        self.pages = {}
        with open(self.data_path + 'page.csv') as f:
            csv_file = csv.reader(f, delimiter=',', quotechar='"')
            for row in csv_file:
                wiki_id, name, wiki_type, _, _ = row
                if(wiki_type in ["0", "2"]):
                   self.pages[int(wiki_id)] = (name, int(wiki_type))
        self._print("--- %.3f seconds ---" % (time.time() - start_time))
        self.pages_reverse = {v[0]: k for k, v in self.pages.items()}
        self._print("Number of pages: {}".format(len(self.pages_reverse)))

    def _load_links(self):
        if(hasattr(self, "links")):
            return
        self._print("Loading links")
        self.links = {}
        start_time = time.time()
        missing_keys = 0
        redirects_counts = 0
        links_count = 0
        with open(self.data_path + 'linkByTarget.csv') as f:
            csv_file = csv.reader(f, delimiter=',', quotechar='"')
            for row in csv_file:
                try:
                    wiki_id = self.pages_reverse[row[0]]
                    wiki_tuple = self.pages[wiki_id]
                    if(wiki_tuple[1] == 2):
                        redirects_counts += 1
                    self.links[wiki_id] = set([int(k) for k in row[1:]])
                    links_count += len(row[1:])
                except KeyError:
                    missing_keys += 1
        self._print("Links count %d" % links_count)
        self._print("Missing keys %d" % missing_keys)
        self._print("Redirects count %d" % redirects_counts)
        self._print("--- %.3f seconds ---" % (time.time() - start_time))


    def _load_occurrences(self):
        if(hasattr(self, "occurrences_counts")):
            return
        self._print("Loading occurrences")
        self.occurrences_counts = {}
        start_time = time.time()
        with open(self.data_path + 'occurrences.tsv') as f:
            for row in f:
                try:
                    row = row.strip()
                    index = row.find(' ')
                    count = int(row[0:index])
                    word = row[index+1:]
                    self.occurrences_counts[word] = count
                except ValueError:
                    pass

        end_time = time.time()
        self._print("Occurrences count: %d" % len(self.occurrences_counts))
        self._print("--- %.3f seconds ---" % (end_time - start_time))

    def _load_counts(self):
        if(hasattr(self, "tokens_counts")):
            return
        self._print("Loading counts")
        self.tokens_counts = {}
        start_time = time.time()
        for fname in ['counts_ngrams.txt', 'counts_unigrams.txt']:
            with open(self.data_path + fname) as f:
                for row in f:
                    try:
                        row = row.rstrip()
                        index = row.rfind(' ')
                        count = float(row[index:])
                        word = row[0:index]
                        self.tokens_counts[word] = count
                    except ValueError:
                        pass

        end_time = time.time()
        self._print("Tokens count: %d" % len(self.tokens_counts))
        self._print("--- %.3f seconds ---" % (end_time - start_time))


    def _load_token_trie(self):
        if(hasattr(self, "trie")):
            return
        self._print("Loading token trie")
        start_time = time.time()
        with open(self.data_path + 'links_trie.pck', 'rb') as f:
            self.trie = pickle.load(f)
        self._print("--- %.3f seconds ---" % (time.time() - start_time))


    def _load_redirects(self):
        if(hasattr(self, "redirects")):
            return
        self._print("Loading redirects")
        self.redirects = {}
        start_time = time.time()
        missing_keys = 0
        with open(self.data_path + 'redirectTargetsBySource.csv') as f:
            csv_file = csv.reader(f, delimiter=',', quotechar='"')
            for row in csv_file:
                try:
                    redirect_id = int(row[0])
                    target_str = row[2]
                    target_id = self.pages_reverse[target_str]
                    self.redirects[redirect_id] = target_id
                except KeyError:
                    missing_keys += 1
        self._print("Redirects count %d" % len(self.redirects))
        end_time = time.time()
        self._print("--- %.3f seconds ---" % (end_time - start_time))

    def _load_senses(self):
        if(hasattr(self, "senses")):
            return
        self._print("Loading senses")
        self.senses = {}
        start_time = time.time()
        missing_senses = 0
        with open(self.data_path + 'links.tsv') as f:
            for row in f:
                try:
                    _, _, _, sense, link = row.rstrip().split('\t')
                    sense = sense[0:1].upper() + sense[1:]
                    try:
                        sense_id = self.pages_reverse[sense]
                        sense_type = self.pages[sense_id][1]
                        if(sense_type == 2):
                            sense_id = self.resolve_redirect(sense_id)
                            sense_type = self.pages[sense_id][1]
                        if(link not in self.senses):
                            self.senses[link] = {}
                        if(sense_id not in self.senses[link]):
                            self.senses[link][sense_id] = 1
                        else:
                            self.senses[link][sense_id] += 1
                    except KeyError:
                        missing_senses += 1
                except ValueError:
                    pass

        end_time = time.time()
        self._print("senses count: %d" % len(self.senses))
        self._print("--- %.3f seconds ---" % (end_time - start_time))

    def _load_concepts(self):
        if(hasattr(self, "concepts")):
            return
        self._print("Loading concepts")
        self.concepts = {}
        start_time = time.time()
        for link, concepts_counts in self.senses.items():
            for concept_id, count in concepts_counts.items():
                if(concept_id not in self.concepts):
                    self.concepts[concept_id] = {}
                self.concepts[concept_id][link] = count
        end_time = time.time()
        self._print("concepts count: %d" % len(self.concepts))
        self._print("--- %.3f seconds ---" % (end_time - start_time))

    def _load_disambiguation_model(self):
        if(hasattr(self, "disambiguation_model")):
            return
        self._print("Loading disambiguation model")
        start_time = time.time()
        self.disambiguation_model = joblib.load(self.data_path + 'disamb-model.forrest.pkl') 
        self._print("--- %.3f seconds ---" % (time.time() - start_time))

    def _load_indices(self):
        if(hasattr(self, "token_index")):
            return
        self._print("Loading indices")
        start_time = time.time()
        self.token_index = {}
        with open(self.data_path + 'tokens.tsv.idx') as tokens_file:
            for row in tokens_file:
                doc_id, pos = [int(e) for e in row.split()]
                self.token_index[doc_id] = pos
        self.link_index = {}
        with open(self.data_path + 'links.tsv.idx') as links_file:
            for row in links_file:
                doc_id, pos = [int(e) for e in row.split()]
                self.link_index[doc_id] = pos
        self._print("--- %.3f seconds ---" % (time.time() - start_time))
        self._print("Number of indexed documents (tokens): {}".format(len(self.token_index)))
        self._print("Number of indexed documents (links): {}".format(len(self.link_index)))

    def _print(self, *args):
        print(*args, file=self.output)
