DVQA - Deep learning-based Video Quality Assessment

## News

- 12/17/2019 add pretrained model on PGC videos

## Installation

We recommend to run the code with virtualenv. The code is developed with Python3.

Please install other prerequisites with the following command after invoking a virtual env.

```
pip install -r requirements.txt
```
All packages are required to run the code.

## Dataset

Please prepare a dataset if you want to evaluate in batch or train the code from scratch on your own GPUs. The dataset should be in json format, e.g. your\_dataset.json

```
{
    "test": {
        "dis": ["dis_1.yuv", "dis_2.yuv"],
        "ref": ["ref_1.yuv", "ref_2.yuv"],
        "fps": [30, 24],
        "mos": [94.2, 55.8],
        "height": [1080, 720],
        "width": [1920, 1280]
    },
    "train": {
        "dis": ["dis_3.yuv", "dis_4.yuv"],
        "ref": ["ref_3.yuv", "ref_4.yuv"],
        "fps": [50, 24],
        "mos": [85.2, 51.8],
        "height": [320, 720],
        "width": [640, 1280]
    }
}
```
For the time being, only YUV is supported. We will update modules to read bitstream.

## Eval a dataset

Put all YUV files (both dis and ref) in a folder and prepare your_dataset.json accordingly. Invoke virtualenv and run:

```
python eval.py --multi_gpu --video_dir /dir/to/yuv --score_file_path /path/to/your_dataset.json --load_model ./save/model_pgc.pt
```

## Train from scratch

Prepare dataset as above and simply run:

```
python train.py --multi_gpu --video_dir /dir/to/yuv --score_file_path /path/to/your_dataset.json --save_model ./save/your_new_trained.pt
```
Please check train.sh and opts.py if you would like to tweak other hyper-parameters.

## Known issues

The pretrained model was trained on 720P PGC videos compressed with H.264/AVC. It runs well with video of a resolution 1920x1080 and below.

We are not sure about the performance when the code is run with the following scenario,
1. PGC with other distortion types, especially time-related distortions.
2. PGC with post-processing filters, like de-nosing, super-resolution, artifacts reduction, etc.
3. UGC videos with pre-processing filter.
4. UGC videos compressed with common codecs.

We will try to answer above questions. Stay tuned.
