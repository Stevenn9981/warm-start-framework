from models.base_recommender import RecommenderBase
from models.mf_joint_numpy import JointMatrixFactorization
import numpy as np
import random
from loguru import logger
from utility.utility import get_combinations


class JointMatrixFactorizaionRecommender(RecommenderBase):
    def __init__(self, split):
        super(JointMatrixFactorizaionRecommender, self).__init__()
        self.split = split
        self.optimal_params = None

    def convert_rating(self, rating):
        if rating == 1:
            return 1
        elif rating == -1:
            return 0
        elif rating == 0:
            # We can make a choice here - either return 0 or 0.5
            return 0.5

    def get_sppmi(self, rating_triples, n_items):
        co_occurrence_matrix = np.zeros((n_items, n_items))

        u_r_map = {}
        for u, e, r in rating_triples:
            if u not in u_r_map:
                u_r_map[u] = []
            if r == 1:
                u_r_map[u].append(e)

        for u, ratings in u_r_map.items():
            for i, r in enumerate(ratings):
                for j, o in enumerate(ratings[i + 1:]):
                    co_occurrence_matrix[r][o] += 1

        D = co_occurrence_matrix.sum()
        hor_sums = co_occurrence_matrix.sum(axis=1).reshape((1, n_items))
        ver_sums = co_occurrence_matrix.sum(axis=0).reshape((1, n_items))

        M = np.divide((co_occurrence_matrix * D), (hor_sums.T @ ver_sums),
                      out=np.zeros_like(co_occurrence_matrix),
                      where=co_occurrence_matrix != 0)

        k = 5
        return M.clip(0, M - np.log(k))

    def batches(self, triples, n=64):
        for i in range(0, len(triples), n):
            yield triples[i:i + n]

    def fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        hit_rates = []

        parameters = {
            'relative_influence': [0.1, 0.25, 0.5, 0.75, 0.9],
            'k': [1, 2, 5, 10, 15, 25, 50]
        }

        if self.optimal_params is None:

            for params in get_combinations(parameters):
                logger.debug(f'Fitting Joint MF with parameters: {params}')
                self.model = JointMatrixFactorization(self.split.n_users, self.split.n_movies + self.split.n_descriptive_entities,
                                     params["k"], params["relative_influence"])

                hit_rates.append((self._fit(training, validation, max_iterations), params))

            hit_rates = sorted(hit_rates, key=lambda x: x[0], reverse=True)
            _, best_params = hit_rates[0]

            self.optimal_params = best_params

            self.model = JointMatrixFactorization(self.split.n_users,
                                                  self.split.n_movies + self.split.n_descriptive_entities,
                                                  self.optimal_params['k'], self.optimal_params['relative_influence'])
            logger.debug(f'Found optimal parameters for Joint MF: {self.optimal_params}')
            self._fit(training, validation)
        else:
            self.model = JointMatrixFactorization(self.split.n_users,
                                                  self.split.n_movies + self.split.n_descriptive_entities,
                                                  self.optimal_params['k'], self.optimal_params['relative_influence'])
            self._fit(training, validation)

    def _fit(self, training, validation, max_iterations=100, verbose=True, save_to='./'):
        # Preprocess training data
        training_triples = []
        for u, ratings in training:
            for r in ratings:
                rating = self.convert_rating(r.rating)
                training_triples.append((u, r.e_idx, rating))

        sppmi = self.get_sppmi(training_triples, self.split.n_entities)

        validation_history = []

        for epoch in range(max_iterations):
            random.shuffle(training_triples)
            self.model.train_als(training_triples, sppmi)

            if epoch % 10 == 0:
                ranks = []
                for user, (pos, negs) in validation:
                    predictions = self.model.predict(user, negs + [pos])
                    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    predictions = {i: rank for rank, (i, s) in enumerate(predictions)}
                    ranks.append(predictions[pos])

                _hit = np.mean([1 if r < 10 else 0 for r in ranks])
                validation_history.append(_hit)

                if verbose:
                    logger.debug(f'Hit@10 at epoch {epoch}: {np.mean([1 if r < 10 else 0 for r in ranks])}')

        return np.mean(validation_history[-10:])

    def predict(self, user, items):
        return self.model.predict(user, items)
