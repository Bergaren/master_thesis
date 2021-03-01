class Config(object):
   def __init__(self):
      self.model = "RNN"

      self.k = 1                       # If 1 don't cluster data. If >1 cluster data into that number of clusters.
      self.embedded_features = False
      self.modes = 1                   # If 1 don't implement VMD. If >1 implement that number of VMD modes.

      self.seasonal_len = 20
      self.weather_len = 24

      self.lookback = 4
      self.output_size = 24*4
      self.mlp_price_len = self.lookback*2 + (24-11)*2
      self.rnn_price_len = 4
      self.input_size = self.configure_input_size()
   
   def configure_input_size(self):
      if self.model == "MLP":
         length = self.mlp_price_len
         print(length)
      elif self.model == "RNN":
         length = self.rnn_price_len
      else:
         length = self.mlp_price_len

      if self.embedded_features:
         length += self.seasonal_len + self.weather_len

      return length
      

