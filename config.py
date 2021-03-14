class Config(object):
   def __init__(self):
      self.model             = "MLP"

      self.embedded_features = False

      self.cluster           = False
      self.k                 = 4 if self.cluster else 1

      self.mode_decomp       = True
      self.ensemmble         = True # Only relevant if mode decomposition is used.
      self.n_modes           = 6 if self.mode_decomp else 1     

      self.lookback = 24
      self.output_size = 24*4
      self.mlp_price_len = self.lookback*2 + (self.lookback-11)*2

      self.seasonal_len = 20
      self.flow_len = self.lookback*17
      self.cap_len = 24*9
      self.market_data = self.cap_len

      self.weather_len = 24

      self.rnn_price_len = 4
      self.input_size = self.configure_input_size()
   
   def configure_input_size(self):
      if self.model == "MLP":
         length = self.mlp_price_len
      elif self.model == "LSTM" or self.model == "GRU":
         length = self.rnn_price_len
      if self.mode_decomp and not self.ensemmble:
         length *= self.n_modes

      if self.embedded_features:
         length += self.market_data
         #length += self.seasonal_len

      return length
      

