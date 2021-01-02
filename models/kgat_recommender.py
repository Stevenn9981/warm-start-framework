import argparse
from loguru import logger
from typing import List, Any, Tuple

from torch import optim
from time import time

from data_loading.generic_data_loader import Rating
# from entrypoint import parser
from models.KGAT import KGAT
from models.base_recommender import RecommenderBase

import os
import random
import collections

import dgl
import torch
import numpy as np
import pandas as pd

from models.trans_h_recommender import convert_ratings


def load_kg_triples(split):
    e_idx_map = split.experiment.dataset.e_idx_map

    with open(split.experiment.dataset.triples_path) as fp:
        df = pd.read_csv(fp)
        triples = [(h, r, t) for h, r, t in df[['head_uri', 'relation', 'tail_uri']].values]
        triples = [(e_idx_map[h], r, e_idx_map[t]) for h, r, t in triples if h in e_idx_map and t in e_idx_map]

    indexed_triples = []
    r_idx_map = {}
    rc = 3
    for h, r, t in triples:
        if r not in r_idx_map:
            r_idx_map[r] = rc
            rc += 1

        # Reverse
        not_r = f'not-{r}'
        if not_r not in r_idx_map:
            r_idx_map[not_r] = rc
            rc += 1

        indexed_triples.append((h, r_idx_map[r], t))
        indexed_triples.append((t, r_idx_map[not_r], h))

    return indexed_triples, r_idx_map


def early_stopping(hit_list, stopping_steps):
    best_hit = max(hit_list)
    best_step = hit_list.index(best_hit)
    if len(hit_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return should_stop


class KGATRecommender(RecommenderBase):
    def __init__(self, split):
        super(KGATRecommender, self).__init__()
        self.split = split
        self.triples_path = split.experiment.dataset.triples_path
        self.entity_idx = split.experiment.dataset.e_idx_map
        self.with_kg_triples = True
        self.seed = 123
        self.n_relations = 3 + 8 + 8
        self.n_users = split.n_users
        self.n_items = split.n_movies
        self.n_entities = split.n_movies + split.n_descriptive_entities
        self.n_users_entities = split.n_users + split.n_movies + split.n_descriptive_entities
        self.cf_batch_size = 1024
        self.kg_batch_size = 2048
        self.if_train = True
        self.optimal_params = None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fit(self, training, validation, max_iterations=1000, verbose=True, save_to='./'):
        user_ratings = convert_ratings(training)
        # Load KG triples if needed
        kg_triples, r_idx_map = load_kg_triples(self.split) if self.with_kg_triples else ([], {})

        # GPU / CPU
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        self.construct_data(kg_triples, user_ratings, if_train=self.if_train)

        item_ids = torch.arange(self.n_items, dtype=torch.long)
        if use_cuda:
            item_ids = item_ids.to(device)
        model = KGAT(self.n_users, self.n_entities, self.n_relations)
        model.to(device)

        logger.info(model)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_graph = self.train_graph
        if not self.if_train:
            test_graph = self.test_graph
        if use_cuda:
            if self.train_graph:
                self.train_graph = self.train_graph.to(device)
                train_graph = self.train_graph
            if not self.if_train and self.test_graph:
                self.test_graph = self.test_graph.to(device)
                test_graph = self.test_graph

        # initialize metrics
        best_epoch = -1
        hr_list = []

        # train model
        for epoch in range(1, max_iterations + 1):
            time0 = time()
            model.train()

            # update attention scores
            with torch.no_grad():
                att = model('calc_att', train_graph)
            train_graph.edata['att'] = att
            logger.info('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            # train cf
            time1 = time()
            cf_total_loss = 0
            n_cf_batch = self.n_cf_train // self.cf_batch_size + 1

            for iter in range(1, n_cf_batch + 1):
                time2 = time()
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.generate_cf_batch(self.train_user_dict)
                if use_cuda:
                    cf_batch_user = cf_batch_user.to(device)
                    cf_batch_pos_item = cf_batch_pos_item.to(device)
                    cf_batch_neg_item = cf_batch_neg_item.to(device)
                cf_batch_loss = model('calc_cf_loss', train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

                cf_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cf_total_loss += cf_batch_loss.item()

                if (iter % 1) == 0:
                    logger.info(
                        'CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                            epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
            logger.info(
                'CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                                  n_cf_batch,
                                                                                                                  time() - time1,
                                                                                                                  cf_total_loss / n_cf_batch))

            # train kg
            time1 = time()
            kg_total_loss = 0
            n_kg_batch = self.n_kg_train // self.kg_batch_size + 1

            for iter in range(1, n_kg_batch + 1):
                time2 = time()
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = self.generate_kg_batch(
                    self.train_kg_dict)
                if use_cuda:
                    kg_batch_head = kg_batch_head.to(device)
                    kg_batch_relation = kg_batch_relation.to(device)
                    kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                    kg_batch_neg_tail = kg_batch_neg_tail.to(device)
                kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail,
                                      kg_batch_neg_tail)

                kg_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                kg_total_loss += kg_batch_loss.item()

                if (iter % 1) == 0:
                    logger.info(
                        'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                            epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss / iter))
            logger.info(
                'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                                  n_kg_batch,
                                                                                                                  time() - time1,
                                                                                                                  kg_total_loss / n_kg_batch))

            logger.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            self.model = model
            # evaluate cf
            hits = 0
            count = 0

            if (epoch % 2) == 0:
                time1 = time()
                model.eval()
                for user, validation_tuple in random.sample(validation, min(len(validation), 200)):
                    count += 1
                    dict = self.predict(user, [validation_tuple[0]] + validation_tuple[1])
                    scores = sorted(dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    if validation_tuple[0] in [item[0] for item in scores]:
                        hits += 1
                logger.info(f'Hit Ratio@10 in Epoch {epoch}: {hits / count * 100:.2f}%')
                logger.info(f'Evaluation time: {time() - time1:.2f}s')
                hr_list.append(hits / count)
                if early_stopping(hr_list, 5):
                    break
            # if (epoch % args.evaluate_every) == 0:
            #     time1 = time()
            #     _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict,
            #                                           user_ids_batches, item_ids, args.K)
            #     logger.info(
            #         'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(
            #             epoch, time() - time1, precision, recall, ndcg))
            #
            #     epoch_list.append(epoch)
            #     precision_list.append(precision)
            #     recall_list.append(recall)
            #     ndcg_list.append(ndcg)
            #     best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)
            #
            #     if should_stop:
            #         break
            #
            #     if recall_list.index(best_recall) == len(recall_list) - 1:
            #         save_model(model, args.save_dir, epoch, best_epoch)
            #         logger.info('Save model on epoch {:04d}!'.format(epoch))
            #         best_epoch = epoch

    def predict(self, user, items):

        user_batch = torch.LongTensor([user + self.n_entities])
        items_batch = torch.LongTensor(items)
        train_graph = self.train_graph

        # GPU / CPU
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self.model
        if use_cuda:
            model.to(device)
            user_batch.to(device)
            items_batch.to(device)
            train_graph.to(device)
        model.eval()
        with torch.no_grad():
            att = model.compute_attention(self.train_graph)
        self.train_graph.edata['att'] = att

        prediction_dict = dict()
        with torch.no_grad():
            cf_scores = model('predict', train_graph, user_batch, items_batch).cpu()
            for i, item_id in enumerate(items):
                prediction_dict[item_id] = cf_scores[0][i].item()
        return prediction_dict

    def construct_data(self, kg_triples, user_ratings: List[Tuple[Any, List[Rating]]], if_train=True):

        # construct kg dict
        if if_train:
            self.train_kg_dict = collections.defaultdict(list)
            self.train_relation_dict = collections.defaultdict(list)
            self.train_user_dict = dict()
            heads = [tri[0] for tri in kg_triples]
            rels = [tri[1] for tri in kg_triples]
            tails = [tri[2] for tri in kg_triples]

            for h, r, t in kg_triples:
                self.train_kg_dict[h].append((t, r))
                self.train_relation_dict[r].append((h, t))
            for uid, ratings in user_ratings:
                h = uid + self.n_entities
                itemids = set()
                for rating in ratings:
                    r = rating.rating
                    t = rating.e_idx
                    heads.append(h)
                    rels.append(r)
                    tails.append(t)
                    if r == 1:
                        itemids.add(t)
                if len(itemids) > 0:
                    self.train_user_dict[h] = list(itemids)

            self.train_graph = dgl.graph((heads, tails), num_nodes=self.n_users_entities)
            self.train_graph.ndata['id'] = torch.arange(self.n_users_entities, dtype=torch.long)
            self.train_graph.edata['type'] = torch.LongTensor(rels)
            self.n_kg_train = len(rels)
            self.n_cf_train = len(rels) - len(kg_triples)

        else:
            self.test_kg_dict = collections.defaultdict(list)
            self.test_relation_dict = collections.defaultdict(list)
            self.test_user_dict = dict()

            heads = [tri[0] for tri in kg_triples]
            rels = [tri[1] for tri in kg_triples]
            tails = [tri[2] for tri in kg_triples]

            for h, r, t in kg_triples:
                self.test_kg_dict[h].append((t, r))
                self.test_relation_dict[r].append((h, t))
            for uid, ratings in user_ratings:
                h = uid + self.n_entities
                itemids = set()
                for rating in ratings:
                    r = rating.rating
                    t = rating.e_idx
                    heads.append(h)
                    rels.append(r)
                    tails.append(t)
                    if r == 1:
                        itemids.add(t)
                if len(itemids) > 0:
                    self.train_user_dict[h] = list(itemids)

            self.test_graph = dgl.graph((heads, tails), num_nodes=self.n_users_entities)
            self.test_graph.ndata['id'] = torch.arange(self.n_users_entities, dtype=torch.long)
            self.test_graph.edata['type'] = torch.LongTensor(rels)
            self.n_kg_test = len(rels)
            self.n_cf_test = len(rels) - len(kg_triples)

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict):
        exist_users = list(user_dict.keys())
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(self.cf_batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail
