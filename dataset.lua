dataset_tools = dofile('coco.lua')
classLabels = dataset_tools.classLabels
numClasses = dataset_tools.numClasses

dataset = torch.load(opts.PATHS.DATASET_CACHED)

dofile('parallel_batch_loader.lua')
dofile('example_loader.lua')
