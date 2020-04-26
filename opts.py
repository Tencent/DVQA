import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/dockerdata/CSIQ_YUV', type=str, help='Path to input videos')
    parser.add_argument('--score_file_path', default='./dataset/csiq_subj_score.json', type=str, help='Path to input subjective score')
    parser.add_argument('--load_model', default='', type=str, help='Path to load checkpoint')
    parser.add_argument('--save_model', default='./save/model_csiq.pt', type=str, help='Path to save checkpoint')
    parser.add_argument('--log_file_name', default='./log/run.log', type=str, help='Path to save log')

    parser.add_argument('--channel', default=1, type=int, help='channel number of input data, 1 for Y channel, 3 for YUV')
    parser.add_argument('--size_x', default=112, type=int, help='patch size x of segment')
    parser.add_argument('--size_y', default=112, type=int, help='patch size y of segment')
    parser.add_argument('--stride_x', default=80, type=int, help='patch stride x between segments')
    parser.add_argument('--stride_y', default=80, type=int, help='patch stride y between segments')

    parser.add_argument('--learning_rate', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='L2 regularization')
    parser.add_argument('--epochs', default=20, type=int, help='epochs to train')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_opts()
    print(args)
