# DOCK
Code for the [DOCK: Detecting Objects by transferring Common-sense Knowledge, ECCV 2018]
Krishna Kumar Singh, Santosh Divvala, Ali Farhadi, Yong Jae Lee
(https://dock-project.github.io/)

If you use our work, please cite it:
```bibtex
@inproceedings{singh-eccv2018,
  title = {DOCK: Detecting Objects by transferring Common-sense Knowledge},
  author = {Krishna Kumar Singh and Yong Jae Lee},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2018}
}

This code is based on the conntextlonet code of Vadim Kantorov (https://github.com/vadimkantorov/contextlocnet)

## Pre-requisites

Please follow the steps of https://github.com/vadimkantorov/contextlocnet

## Pre-trained Models

You can find the pre-trained models here: https://drive.google.com/open?id=1QobgtcKX06QOY70Og3ezoRm_1Gns5112

## Test Model

```
th test.lua <pre-trained model>  <unique_id>
```

The above command will apply pre-trained model on all the MS COCO test images and gives class scores for each proposals. You can use unique_id to specify name of file which would be used to save the test scores.

## More code coming soon

