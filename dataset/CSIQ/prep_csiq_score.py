import os
import re
from collections import OrderedDict
import numpy as np
import json


def make_score_file():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    seq_file_name = 'csiq_video_quality_seqs.txt'
    score_file_name = 'csiq_video_quality_data.txt'

    all_scenes = ['Keiba', 'Timelapse', 'BQTerrace', 'Carving', 'Chipmunks',
                  'Flowervase', 'ParkScene', 'PartyScene', 'BQMall', 'Cactus',
                  'Kimono','BasketballDrive']
    test_scene = ['BQTerrace', 'ParkScene']
    framerate = {
        'Chipmunks_832x480_ref.yuv': 24,
        'Kimono_832x480_ref.yuv': 24,
        'ParkScene_832x480_ref.yuv': 24,
        'Carving_832x480_ref.yuv': 25,
        'Flowervase_832x480_ref.yuv': 30,
        'Keiba_832x480_ref.yuv': 30,
        'Timelapse_832x480_ref.yuv': 30,
        'BasketballDrive_832x480_ref.yuv': 50,
        'Cactus_832x480_ref.yuv': 50,
        'PartyScene_832x480_ref.yuv': 50,
        'BQMall_832x480_ref.yuv': 60,
        'BQTerrace_832x480_ref.yuv': 60}

    width = 832
    height = 480

    seqs = np.genfromtxt(os.path.join(dir_path, seq_file_name), dtype='str')
    score = np.genfromtxt(os.path.join(dir_path, score_file_name), dtype='float')

    ret = OrderedDict()
    ret['train'] = OrderedDict()
    ret['test'] = OrderedDict()

    trn_dis = []
    trn_ref = []
    trn_mos = []
    trn_height = []
    trn_width = []
    trn_fps = []

    tst_dis = []
    tst_ref = []
    tst_mos = []
    tst_height = []
    tst_width = []
    tst_fps = []

    for clip, mos in zip(seqs, score):
        clip_info = re.split('_', clip)
        clip_name = clip_info[0]
        ref = clip.replace(clip[-10:-4], 'ref')
        fps = framerate[ref]

        if clip_name in test_scene:
            tst_dis.append(clip)
            tst_ref.append(ref)
            tst_mos.append(100.0 - float(mos[0]))
            tst_height.append(height)
            tst_width.append(width)
            tst_fps.append(fps)
        else:
            trn_dis.append(clip)
            trn_ref.append(ref)
            trn_mos.append(100.0 - float(mos[0]))
            trn_height.append(height)
            trn_width.append(width)
            trn_fps.append(fps)

    ret['train']['dis'] = trn_dis
    ret['train']['ref'] = trn_ref
    ret['train']['mos'] = trn_mos
    ret['train']['height'] = trn_height
    ret['train']['width'] = trn_width
    ret['train']['fps'] = trn_fps

    ret['test']['dis'] = tst_dis
    ret['test']['ref'] = tst_ref
    ret['test']['mos'] = tst_mos
    ret['test']['height'] = tst_height
    ret['test']['width'] = tst_width
    ret['test']['fps'] = tst_fps

    with open('csiq_subj_score_{}.json'.format('_'.join(test_scene)), 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    make_score_file()

    print('Done')
