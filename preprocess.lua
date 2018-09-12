require 'cudnn'
require 'loadcaffe'
require 'image'
require 'string'
matio = require 'matio'
voc_tools = dofile('coco.lua')

dofile('opts.lua')

function VGGF()
	local model_converted = loadcaffe.load(opts.PATHS.BASE_MODEL_RAW.PROTOTXT, opts.PATHS.BASE_MODEL_RAW.CAFFEMODEL, 'cudnn'):float()
	torch.save(opts.PATHS.BASE_MODEL_CACHED, model_converted)
end

function VOC()
	local function copy_proposals_in_dataset(trainval_test_mat_paths, voc)
		local subset_paths = {{'train', trainval_test_mat_paths.trainval}, {'val', trainval_test_mat_paths.trainval}}

		local m = {train = {}, val = {}}
		local b = {train = nil, val = nil}
		local s = {train = nil, val = nil}
		for _, t in ipairs(subset_paths) do
			annFile = '/mnt/raid01/krishnas/coco/annotations/instances_'..t[1]..'2014.json';
                        cocoApi=coco.CocoApi(annFile)
                        imgIds = cocoApi:getImgIds()
			b[t[1]]={};
			s[t[1]]={};
                        for im_itr=1,imgIds:numel() do
				img = cocoApi:loadImgs(imgIds[im_itr])[1]
				mat_name=string.gsub(img.file_name,'jpg','mat');
				local h = matio.load('/mnt/raid01/krishnas/proposals/MCG-COCO-'..t[1]..'2014-boxes/'..mat_name)
				b[t[1]][im_itr]=h.boxes;	
				s[t[1]][im_itr]=h.scores;	
			end
		end

		for _, subset in ipairs{'train', 'val'} do
			voc[subset].rois = {}
			for exampleIdx = 1, voc[subset]:getNumExamples() do
				print('read:'..exampleIdx); 
				local ind = exampleIdx;
				local box_scores = s[subset] and s[subset][ind] or torch.FloatTensor(b[subset][ind]:size(1), 1):zero()
				voc[subset].rois[exampleIdx] = torch.cat(b[subset][ind]:index(2, torch.LongTensor{2, 1, 4, 3}):float() - 1, box_scores:float())

				if s[subset] then
					voc[subset].rois[exampleIdx] = voc[subset].rois[exampleIdx]:index(1, ({box_scores:squeeze(2):sort(1, true)})[2]:sub(1, math.min(box_scores:size(1), 2048)))
				end
			end
			voc[subset].getProposals = function(self, exampleIdx)
				return self.rois[exampleIdx]
			end
		end
	end

	local function filter_proposals(voc)
		local min_width_height = 10
		for _, subset in ipairs{'train', 'val'} do
			for exampleIdx = 1, voc[subset]:getNumExamples() do
				print('check:'..exampleIdx); 
				local x1, y1, x2, y2 = unpack(voc[subset].rois[exampleIdx]:split(1, 2))
				local channels, height, width = unpack(image.decompressJPG(voc[subset]:getJpegBytes(exampleIdx)):size():totable())
				
				assert(x1:ge(0):all() and x1:le(width):all())
				assert(x2:ge(0):all() and x2:le(width):all())
				assert(y1:ge(0):all() and y1:le(height):all())
				assert(y2:ge(0):all() and y2:le(height):all())
				assert(x1:le(x2):all() and y1:le(y2):all())

				voc[subset].rois[exampleIdx] = voc[subset].rois[exampleIdx]:index(1, (x2 - x1):ge(min_width_height):cmul((y2 - y1):ge(min_width_height)):squeeze(2):nonzero():squeeze(2))
			end
		end
	end

	local voc = voc_tools.load(opts.PATHS.VOC_DEVKIT_VOCYEAR)
	copy_proposals_in_dataset(opts.PATHS.PROPOSALS, voc)
	filter_proposals(voc)
	torch.save(opts.PATHS.DATASET_CACHED, voc)
end

for _, a in ipairs(arg) do
	print('Preprocessing', a)
	_G[a]()
end
print('Done')
