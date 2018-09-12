model = nn.Sequential():
	add(nn.ParallelTable():
	add(nn.Sequential():
	add(nn.ParallelTable():
		add(base_model.conv_layers):
		add(nn.Identity())
	):
	add(RectangularRingRoiPooling(base_model.pooled_height, base_model.pooled_width, base_model.spatial_scale, base_model.spp_correction_params)):
	add(RoiReshaper:StoreShape()):
	add(base_model.fc_layers_view(RoiReshaper)):
	add(base_model.fc_layers):
	add(nn.ConcatTable():
		add(nn.Sequential():
			add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8c')):
			add(RoiReshaper:RestoreShape()):
			add(nn.Squeeze(1)):
			add(cudnn.SoftMax()):
			add(nn.Unsqueeze(1)):
			named('output_fc8c')
		):
		add(nn.Sequential():
			add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8d')):
			add(RoiReshaper:RestoreShape(4)):
			add(cudnn.SpatialSoftMax()):
			add(nn.Squeeze(4)):
			named('output_softmax')
		)
	):
	add(nn.CMulTable():named('output_prod_old'))):
	add(nn.Unsqueeze(1))):
	add(nn.CMulTable():named('output_prod')):
	add(nn.Sum(2))
print(numClasses)
classification_criterion = nn.BCECriterion(nil, false)
classification_criterion.updateOutput = function(self, input, target) return nn.BCECriterion.updateOutput(self, input, target * 0.5 + 0.5) end
classification_criterion.updateGradInput = function(self, input, target) return nn.BCECriterion.updateGradInput(self, input, target * 0.5 + 0.5) end
criterion = classification_criterion
optimState = {learningRate = 1e-5, momentum = 0.9, weightDecay = 5e-4}
optimState_annealed = {learningRate = 1e-6, momentum = 0.9, weightDecay = 5e-4, epoch = 20}
