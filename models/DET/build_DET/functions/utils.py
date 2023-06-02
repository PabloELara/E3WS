import matplotlib.pyplot as plt
import numpy as np

#This class return probability of being Earthquake
class pb_SAM():
	def predict(self, X):
		meta_features = np.column_stack([
			np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
			for base_models in self.base_models_ ])
		return self.meta_model_.predict(meta_features)

