import numpy as np

class Config(object):
   def __init__(self):
      self.model              = "MLP"

      self.embedded_features  = True

      self.cluster            = False
      self.k                  = 4 if self.cluster else 1

      self.mode_decomp        = False
      self.ensemble           = False # Only relevant if mode decomposition is used.
      self.n_modes            = 6 if self.mode_decomp else 1     

      self.min_lookback       = 1 if self.model in ['LSTM', 'GRU', 'THD'] else 12
      self.max_lookback       = 30 if self.model in ['LSTM', 'GRU', 'THD'] else 24*2
      self.output_size        = 24

      self.prod_len           = 24*3
      self.cons_len           = 24
      self.cap_len            = 24*5
      self.seasonal_len       = 62
      self.flow_len           = 5 if self.model in ['LSTM', 'GRU', 'THD'] else self.max_lookback*5

      self.price_dayahead_len          = 1 if self.model in ['LSTM', 'GRU', 'THD'] else self.max_lookback
      self.price_intraday_len          = 1 if self.model in ['LSTM', 'GRU', 'THD'] else (self.max_lookback-11)
      self.price_len                   = self.price_dayahead_len + self.price_intraday_len
      self.input_size         = self.configure_input_size()
      self.f_mapping          = self.construct_feature_mapping()
   
   def configure_input_size(self):
      length = self.price_len

      if self.embedded_features:
         length += self.prod_len + self.cap_len + self.cons_len + self.seasonal_len + self.flow_len
      return length

   def construct_feature_mapping(self):
      f_idxs = {
         'price dayahead'        : np.array(range(self.price_dayahead_len)),
         'price intraday'        : np.array(range(self.price_intraday_len)),
         'wp'           : np.array(range(1)),
         'nuclear'      : np.array(range(1)),
         'solar'        : np.array(range(1)),
         'cons'         : np.array(range(1)),
         'cap'          : np.array(range(self.cap_len)),
         'year'         : np.array(range(1)),
         'week'         : np.array(range(53)),
         'day of week'  : np.array(range(7)),
         'holiday'      : np.array(range(1)),
         'flow'         : np.array(range(self.flow_len))
      }

      start = 0
      for k, v in f_idxs.items():
         f_idxs[k] = np.add(v, start)
         start = f_idxs[k][-1] + 1
      
      return f_idxs



      

