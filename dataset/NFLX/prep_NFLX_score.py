import os
import re
from collections import OrderedDict
import numpy as np
import json
from NFLX_dataset_public import dis_videos

def make_score_file():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    all_scenes = ['BigBuckBunny', 'BirdsInCage', 'CrowdRun', 'ElFuente1', 'ElFuente2',
                  'FoxBird', 'OldTownCross', 'Seeking', 'Tennis']
    test_scene = all_scenes
    framerate = {
        'BigBuckBunny_25fps.yuv': 25,
        'BirdsInCage_30fps.yuv': 30,
        'CrowdRun_25fps.yuv': 25,
        'ElFuente1_30fps.yuv': 30,
        'ElFuente2_30fps.yuv': 30,
        'FoxBird_25fps.yuv': 25,
        'OldTownCross_25fps.yuv': 25,
        'Seeking_25fps.yuv': 25,
        'Tennis_24fps.yuv': 24}

    width = 1920
    height = 1080

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

    for pvs in dis_videos:

        clip_info = re.split('/', pvs['path'])
        clip_name = clip_info[-1]
        print(clip_name)
        scene = re.split('_', clip_name)[0]
        for k, v in framerate.items():
            if scene in k:
                if scene in test_scene:
                    tst_dis.append(clip_name)
                    tst_ref.append(k)
                    tst_mos.append(pvs['dmos'])
                    tst_height.append(height)
                    tst_width.append(width)
                    tst_fps.append(v)
                else:
                    trn_dis.append(clip_name)
                    trn_ref.append(k)
                    trn_mos.append(pvs['dmos'])
                    trn_height.append(height)
                    trn_width.append(width)
                    trn_fps.append(v)
                break

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

    with open('NFLX_subj_score.json', 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    make_score_file()

    print('Done')
