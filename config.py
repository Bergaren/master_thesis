import numpy as np
import seaborn as sns


class Config(object):
    def __init__(self):
        self.model = "THH"
        self.rnn_head = "GRU"
        self.ensemble = True
        self.regression = True
        self.name = self.get_full_name()

        self.embedded_features = True

        self.n_models = 4 if self.ensemble else 1

        self.min_lookback = 1
        self.max_lookback = 30 if self.model in ['LSTM', 'GRU', 'THH'] else 5
        self.output_size = 24

        self.prod_len = 24*3
        self.cons_len = 24
        self.cap_len = 24*5
        self.seasonal_len = 62
        self.flow_len = 1 if self.model in [
            'LSTM', 'GRU', 'THH'] else self.max_lookback*24

        self.price_dayahead_len = 1 if self.model in [
            'LSTM', 'GRU', 'THH'] else self.max_lookback*24
        self.price_intraday_len = 1 if self.model in [
            'LSTM', 'GRU', 'THH'] else (self.max_lookback*24-11)
        self.price_len = self.price_dayahead_len + self.price_intraday_len
        self.input_size = self.configure_input_size()
        self.f_mapping = self.construct_feature_map()

        self.patience = 10

        self.colors = sns.color_palette("tab10")

    def configure_input_size(self):
        length = self.price_len

        if self.embedded_features:
            length += self.prod_len + self.cap_len + \
                self.cons_len + self.seasonal_len + self.flow_len
        return length

    def construct_feature_map(self):
        f_idxs = {
            'price dayahead': np.array(range(self.price_dayahead_len)),
            'price intraday': np.array(range(self.price_intraday_len)),
            'wp': np.array(range(1)),
            'nuclear': np.array(range(1)),
            'solar': np.array(range(1)),
            'cons': np.array(range(1)),
            'cap': np.array(range(self.cap_len)),
            'year': np.array(range(1)),
            'week': np.array(range(53)),
            'day of week': np.array(range(7)),
            'holiday': np.array(range(1)),
            'flow 1': np.array(range(self.flow_len)),
            'flow 2': np.array(range(self.flow_len)),
            'flow 3': np.array(range(self.flow_len)),
            'flow 4': np.array(range(self.flow_len)),
            'flow 5': np.array(range(self.flow_len))
        }

        start = 0
        for k, v in f_idxs.items():
            f_idxs[k] = np.add(v, start)
            start = f_idxs[k][-1] + 1

        return f_idxs

    def get_full_name(self):
        name = self.model
        if name == 'THH':
            name += '_' + self.rnn_head
        if self.ensemble:
            name += '_Ensemble'
        if not self.regression:
            name += '_Classification_network'
        return name

    def get_model_path(self):
        return 'saved_models/{}'.format(self.name)

    def get_param_path(self, *args):
        if self.ensemble:
            prefix = chr(ord('@')+args[0]+1)
            return 'saved_models/{}_{}_params.txt'.format(prefix, self.name)
        else:
            return 'saved_models/{}_params.txt'.format(self.name)

    def get_results_path(self, test_type, ext):
        return 'results/{}_{}.{}'.format(test_type, self.name, ext)
