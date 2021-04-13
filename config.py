class Config(object):
   def __init__(self):
      self.model             = "MLP"

      self.embedded_features = True

      self.cluster           = False
      self.k                 = 4 if self.cluster else 1

      self.mode_decomp       = False
      self.ensemble          = False # Only relevant if mode decomposition is used.
      self.n_modes           = 6 if self.mode_decomp else 1     

      self.lookback = 24
      self.output_size = 24
      self.mlp_price_len = self.lookback + (self.lookback-11)

      self.seasonal_len = 20
      self.cap_len = 24*5

      self.rnn_price_len = 2
      self.input_size = self.configure_input_size()
   
   def configure_input_size(self):
      if self.model == "MLP":
         length = self.mlp_price_len
      elif self.model == "LSTM" or self.model == "GRU":
         length = self.rnn_price_len
      if self.mode_decomp and not self.ensemble:
         length *= (self.n_modes + 1)

      if self.embedded_features:
         #length += self.cap_len
         length += self.seasonal_len

      return length
      

