# VSSVM
SVM for Video Splitting

## Dataset Preparation
You need to prepare the dataset first. You can use the `prepare_dataset.py` script to extract the features from the video and save them to a file.
We privide the weight of the svm model in the 'svm_model.pkl'. You can use the `train_svm.py` script to train the svm model.
#### Usage
```bash
python prepare_dataset.py --video_path <video_path> --output_path <output_path> --numsamples <numsamples>

python train_svm.py --video_path <video_path>

python DetectScene.py --video_path <video_path> --model_path <model_path>
```

#### Reference
[1] [Koala-36M: A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content][https://arxiv.org/pdf/2410.08260]
