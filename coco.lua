coco = require 'coco'
local classLabels = torch.load('/mnt/raid01/krishnas/coco/categories.t7');


return {
	classLabels = classLabels,
	numClasses = #classLabels,

	load = function()
		local xml = require 'xml'


		local numMaxExamples = 90000
		local numMaxObjectsPerExample = 100
		local mkDataset = function(strsize,datype) return 
		{
			filenames = torch.CharTensor(numMaxExamples, strsize):zero(),
			labels = torch.FloatTensor(numMaxExamples, #classLabels):fill(-1),
			objectBoxes = torch.FloatTensor(numMaxExamples * numMaxObjectsPerExample, 5):zero(),
			objectBoxesInds = torch.IntTensor(numMaxExamples, 2):zero(),
			jpegs = torch.ByteTensor(numMaxExamples):zero(),
			jpegsInds = torch.LongTensor(numMaxExamples, 2):zero(),
			mydatype=datype,
			getNumExamples = function(self)
				return self.numExamples
			end,

			getImageFileName = function(self, exampleIdx)
				return self.filenames[exampleIdx]:clone():storage():string():match('%Z+')
			end,

			getGroundTruthBoxes = function(self, exampleIdx)
				return self.objectBoxes:sub(self.objectBoxesInds[exampleIdx][1], self.objectBoxesInds[exampleIdx][2])
			end,

			getJpegBytes = function(self, exampleIdx)
				self.f = torch.DiskFile(paths.concat('/mnt/raid01/krishnas/coco/coco_images/',self.mydatype..'2014/', self.filenames[exampleIdx]:clone():storage():string():match('%Z+')), 'r')
				self.f:binary()
				self.f:seekEnd()
				self.file_size_bytes = self.f:position() - 1
				self.f:seek(1)
				self.bytes = torch.ByteTensor(self.file_size_bytes)
				self.f:readByte(self.bytes:storage())
				self.f:close()
				self.f=1
				return self.bytes
			end,

			getLabels = function(self, exampleIdx)
				return self.labels[exampleIdx]
			end
		} end

		local voc = { train = mkDataset(31,'train'), val = mkDataset(29,'val')}

		for _, subset in ipairs{'train','val'} do
			
			print (classLabels)
			local exampleIdx = 1
			local jpegsFirstByteInd = 1
			annFile = '/mnt/raid01/krishnas/coco/annotations/instances_'..subset..'2014.json';
			cocoApi=coco.CocoApi(annFile)
			imgIds = cocoApi:getImgIds()
			for im_itr=1,imgIds:numel() do
				assert(exampleIdx <= numMaxExamples)
				img = cocoApi:loadImgs(imgIds[im_itr])[1]
				voc[subset].filenames[exampleIdx]=torch.CharTensor(torch.CharStorage():string(img.file_name));
				print(img.file_name);	
				exampleIdx = exampleIdx + 1
			end
			voc[subset].numExamples = exampleIdx - 1
		end	 
		local testHasAnnotation = false
		local objectBoxIdx = 1
		for _, subset in ipairs(testHasAnnotation and {'train', 'val', 'test'} or {'train', 'val'})  do
			annFile = '/mnt/raid01/krishnas/coco/annotations/instances_'..subset..'2014.json';
			cocoApi=coco.CocoApi(annFile)
			imgIds = cocoApi:getImgIds()
			for im_itr=1,imgIds:numel() do
				imgId= imgIds[im_itr];
				annIds = cocoApi:getAnnIds({imgId=imgId});
				anns = cocoApi:loadAnns(annIds)
				local firstObjectBoxIdx = objectBoxIdx
				for ob_itr=1,#anns do
					print(anns[ob_itr].category_idx)
					print(voc[subset].labels:size())
					voc[subset].labels[im_itr][anns[ob_itr].category_idx]=1;
					local xmin = anns[ob_itr].bbox[1];
					local xmax = anns[ob_itr].bbox[1]+anns[ob_itr].bbox[3];
					local ymin = anns[ob_itr].bbox[2];
					local ymax = anns[ob_itr].bbox[2]+anns[ob_itr].bbox[4];
					local classLabelInd = anns[ob_itr].category_idx;
					voc[subset].objectBoxes[objectBoxIdx] = torch.FloatTensor({classLabelInd, xmin, ymin, xmax, ymax})
					objectBoxIdx = objectBoxIdx + 1
				end
				voc[subset].objectBoxesInds[im_itr] = torch.IntTensor({firstObjectBoxIdx, objectBoxIdx - 1})
			end
		end
		for _, subset in ipairs{'train', 'val'} do
			voc[subset].filenames = voc[subset].filenames:sub(1, voc[subset].numExamples):clone()
			voc[subset].labels = voc[subset].labels:sub(1, voc[subset].numExamples):clone()
			voc[subset].jpegsInds = voc[subset].jpegsInds:sub(1, voc[subset].numExamples):clone()

			if voc[subset].objectBoxes and voc[subset].objectBoxesInds then
				voc[subset].objectBoxesInds =  voc[subset].objectBoxesInds:sub(1, voc[subset].numExamples):clone()
				voc[subset].objectBoxes = voc[subset].objectBoxes:sub(1, voc[subset].objectBoxesInds[voc[subset].numExamples][2]):clone()
			end
		end
		return voc
	end,

	package_submission = function(OUT, voc, fol_name, subset, task, ...)
		local task_a, task_b  = task:match('(.+)_(.+)')
		local write = {
			cls = function(f, classLabelInd, scores)
				assert(voc[subset]:getNumExamples() == scores:size(1))

				for exampleIdx = 1, voc[subset]:getNumExamples() do
					f:write(string.format('%s %.12f\n', voc[subset]:getImageFileName(exampleIdx), scores[exampleIdx][classLabelInd]))
				end
			end,
			det = function(f, classLabelInd,myimid,myclassid, rois, scores, mask)

				for exampleIdx = 1, voc[subset]:getNumExamples() do
					for roiInd = 1, scores[exampleIdx]:size(scores[exampleIdx]:dim()) do
						if mask[exampleIdx][classLabelInd][roiInd] > 0 then
							f:write(string.format('{\n"image_id" :%d,\n "category_id" : %d,\n "bbox" : [ %.12f, %.12f, %.12f, %.12f],\n "score" : %.12f\n},\n ',
								myimid[exampleIdx],
								myclassid[classLabelInd], 
								math.max(0, rois[exampleIdx][roiInd][1] ),
								math.max(0, rois[exampleIdx][roiInd][2] ),
								math.max(0, rois[exampleIdx][roiInd][3] )-math.max(0, rois[exampleIdx][roiInd][1] ),
								math.max(0, rois[exampleIdx][roiInd][4] )- math.max(0, rois[exampleIdx][roiInd][2] ),
								scores[exampleIdx][classLabelInd][roiInd]
							))
						end
					end
				end
			end
		}

		os.execute(string.format('mkdir -p "%s/results/%s/Main"', '/mnt/raid01/krishnas/mymodels/output/', fol_name))

		local respath = string.format('%s/results/%s/Main/%%s_%s_%s_%%s.txt', '/mnt/raid01/krishnas/mymodels/output/', fol_name, task_b, subset)

		threads = require 'threads'
		threads.Threads.serialization('threads.sharedserialize')
		jobQueue = threads.Threads(5)
		local writer = write[task_b]
		annFile = '/mnt/raid01/krishnas/coco/annotations/instances_'..subset..'2014.json';
		cocoApi=coco.CocoApi(annFile)
		local imgIds = cocoApi:getImgIds()
		local classIds = cocoApi:getCatIds()
		print(imgIds)
		for classLabelInd, classLabel in ipairs(classLabels) do
		
			jobQueue:addjob(function(...)
				local f = assert(io.open(respath:format(task_a, classLabel), 'w'))
				writer(f, classLabelInd,imgIds,classIds, ...)
				f:close()
			end, function() end, ...)
		end
		jobQueue:synchronize()
		os.execute(string.format('cd "%s" && tar -czf "results-%s-%s-%s.tar.gz" results', OUT, fol_name, task, subset))
		return respath
	end,

	vis_classification_submission = function(OUT, fol_name, subset, classLabel, JPEGImages_DIR, top_k)
		top_k = top_k or 20
		local res_file_path = string.format('%s/results/%s/Main/comp2_cls_%s_%s.txt', OUT, fol_name, subset, classLabel)

		local scores = {}
		for line in assert(io.open(res_file_path)):lines() do
			scores[#scores + 1] = line:split(' ')
		end

		table.sort(scores, function(a, b) return -tonumber(a[2]) < -tonumber(b[2]) end)

		local image = require 'image'
		local top_imgs = {}
		print('K = ', top_k)
		for i = 1, top_k do
			top_imgs[i] = image.scale(image.load(paths.concat(JPEGImages_DIR, scores[i][1] .. '.jpg')), 128, 128)
			print(scores[i][2], scores[i][1])
		end

		image.display(top_imgs)
	end,
	
	precisionrecall = precisionrecall,

	meanAP = function(scores_all, labels_all)
		return ({precisionrecall(scores_all, labels_all)})[3]:mean()
	end
}
