#!/usr/bin/env python
import sys
import os.path
import time
from wiki_dsmb.concept_block_generator import ConceptBlockGenerator
from wiki_dsmb.trie_matcher import TrieMatcher
from wiki_dsmb.segment_selector import SegmentSelector

class Disambiguator:
    rejected_words = set(["ref", "code", "syntaxhighlight", "name", "math", "/math"])

    def __init__(self, data_path, wiki, min_link_prob=0.005, min_model_prob=0.5, min_concept_block=5, max_concept_block=50, output_file=sys.stdout):
        self.data_path = data_path 
        self.concept_generator = ConceptBlockGenerator(data_path + 'links.tsv', wiki, min_concept_block, max_concept_block)
        matcher = TrieMatcher(data_path + 'tokens.tsv', wiki)
        self.segment_selector = SegmentSelector(matcher, wiki)
        self.wiki = wiki
        self.output_file = output_file
        self.min_link_prob = min_link_prob
        self.min_model_prob = min_model_prob

    def disambiguate(self, first_doc=0, last_doc=10):
        self.concept_blocks = self.concept_generator.blocks(first_doc, last_doc)
        self.selected_segments = self.segment_selector.selected_links(first_doc, last_doc)
        result_path = self.data_path + "dsmb/wiki.dsmb.{}-{}.txt".format(first_doc, last_doc)
        time_path = self.data_path + "dsmb/wiki.time.{}-{}.txt".format(first_doc, last_doc)
        if(os.path.isfile(result_path)):
            self._print("The file {} already exists. Remove it in order to proceed.".format(result_path))
            return
        self.wiki.load()
        self.disambiguation_model = self.wiki.disambiguation_model
        self.result_file = open(result_path, 'w')
        self.time_file = open(time_path, 'w')
        self.time_file.write("time\tlinks/s\tconcepts/s\n")
        last_time = time.time()
        while(True):
            try:
                existing_links, weights, goodness, next_token_index, doc_id = next(self.concept_blocks)
                doc_id = int(doc_id)
                # skip pages that are not regular pages (e.g. redirects, disambiguation pages, etc.)
                if(doc_id not in self.wiki.pages):
                  continue
                self._print("=== [%i] Concept block len:%i token:%i doc:%s ===" % (doc_id, len(existing_links),
                                                                              next_token_index, self.wiki.pages[doc_id][0]))
                weights, goodness = self.wiki.compute_weights_and_goodness(existing_links)
                detected_links = set()
                segments = []
                disambiguated_links = {c[0] : c[1] for c in existing_links}
                while(True):
                    segment = next(self.selected_segments)
                    if(segment.document_id < doc_id):
                        continue
                    if(segment.last_token_index >= next_token_index or segment.document_id > doc_id):
                        self.segment_selector.push_back()
                        break
                    if(segment.string in self.rejected_words):
                        segment.string = " "
                    segments.append(segment)
                    l_prob = self.wiki.link_probability(segment.string)
                    if(l_prob < self.min_link_prob):
                        continue
                    if(segment.string not in disambiguated_links):
                        detected_links.add(segment.string)
                self._print(len(detected_links))
                if(len(detected_links) == 0):
                  continue
                features, candidates = self.wiki.compute_features(detected_links, weights, goodness, existing_links)
                if(len(features) == 0):
                  continue
                #self._print("l_prob, goodne, s_prob, s_rank, c_prob, c_rank, sem_rel, sem_rank")
                probabilities = self.disambiguation_model.predict_proba(features)
                offset = 0
                for link_candidate in candidates.keys():
                    if(len(candidates[link_candidate]) == 0):
                        continue
                    candidate_senses = candidates[link_candidate]
                    offset_end = (offset+len(candidate_senses))
                    max_probability = max(probabilities[offset:offset_end], key=lambda s: s[1])[1]
                    selected_indices = []
      
                    for candidate_index in range(len(candidate_senses)):
                        #self._print("{} {}".format(pages[candidate_senses[candidate_index]][0], probabilities[offset + candidate_index]))
                        #self._print(" ".join(["{:f}".format(e) for e in features[offset + candidate_index]]))
                        if(probabilities[offset + candidate_index][1] == max_probability):
                            selected_indices.append(candidate_index)
      
                    best_rank = min([features[offset + fi][-1] for fi in range(len(candidate_senses)) if fi in selected_indices])
                    best_id = [candidate_senses[si] for si in range(len(candidate_senses)) if si in selected_indices and features[offset+si][-1] == best_rank][0]
                    if(max_probability > self.min_model_prob):
                        #self._print("{:25} -> {} {}".format(link_candidate, pages[best_id][0], max_probability))
                        disambiguated_links[link_candidate] = best_id
                    offset = offset_end
      
                for segment in segments:
                    if(segment.string in disambiguated_links):
                        file_output = " ___{}___{}___ ".format(segment.string.replace(" ", "_"), disambiguated_links[segment.string])
                        #output = segment.string.replace(" ", "_") + ":[" + pages[disambiguated_links[segment.string]][0] + "]"
                    else:
                        file_output = segment.string
                        #output = segment.string
                    if(segment.space == "1"):
                        file_output = " " + file_output
                        #output = " " + output
                    #self._print(output, end='')
                    self.result_file.write(file_output)
                #self._print()
                self.result_file.write("\n")
                self.result_file.flush()
                time_diff = time.time() - last_time
                links_ps = time_diff / len(detected_links)
                concepts_ps = time_diff / len(existing_links)
                self.time_file.write("%.1f\t%.6f\t%.6f\n" % (time_diff, links_ps, concepts_ps))
                self.time_file.flush()
                last_time = time.time() 
            except StopIteration:
                break
        self.result_file.close()

    def _print(self, *args):
        print(*args, file=self.output_file)



if __name__=='__main__':
    from wiki_dsmb.wiki import Wiki
    from wiki_dsmb.trie_element import TrieElement
    from wiki_dsmb.token_trie import TokenTrie
    import argparse

    parser = argparse.ArgumentParser(description='Wikipedia disambiguator.')
    parser.add_argument('--data_path', required=True, help='the path to the directory with Wikipedia data')
    parser.add_argument('--first_doc', required=True, type=int, default=1, help='the id of the first document to disambiguate')
    parser.add_argument('--last_doc', required=True, type=int, default=10, help='the id of the first document *not to* to disambiguate')

    args = parser.parse_args()

    MIN_LINK_PROB = 0.005
    MAX_CONCEPT_BLOCK = 50
    MIN_CONCEPT_BLOCK = 5
    MIN_SENSE_PROB = 0.01
    MIN_MODEL_PROB = 0.5

    #data_path = '/net/scratch/people/plgapohl/wiki/pl/2017/'
    wiki = Wiki(args.data_path, MIN_SENSE_PROB)
    disambiguator = Disambiguator(args.data_path, wiki, MIN_LINK_PROB, MIN_MODEL_PROB, MIN_CONCEPT_BLOCK, MAX_CONCEPT_BLOCK)
    disambiguator.disambiguate(args.first_doc, args.last_doc)
