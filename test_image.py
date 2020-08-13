from SPADE import SPADE
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of SPADE"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--test_file_path', type=str, help='test file path for single image test')
    parser.add_argument('--dataset', type=str, default='AED20K', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    # The total number of iterations is [epoch * iteration]

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=5000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--TTUR', type=str2bool, default=True, help='Use TTUR training scheme')

    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--vgg_weight', type=int, default=10, help='Weight about perceptual loss')
    parser.add_argument('--feature_weight', type=int, default=10, help='Weight about discriminator feature matching loss')
    parser.add_argument('--kl_weight', type=float, default=0.05, help='Weight about kl-divergence')

    parser.add_argument('--gan_type', type=str, default='hinge', help='gan / lsgan / hinge / wgan-gp / wgan-lp / dragan')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')

    parser.add_argument('--n_dis', type=int, default=4, help='The number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=2, help='number of scales')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')

    parser.add_argument('--num_upsampling_layers', type=str, default='more',
                        choices=('normal', 'more', 'most'),
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. "
                             "If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--img_height', type=int, default=256, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--segmap_ch', type=int, default=3, help='The size of segmap channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return parser.parse_args()



def main():
    # parse arguments
    args = parse_args()
    print(args)
    if args is None:
      print("no args")
      exit()

    # open session
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        gan = SPADE(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        gan.single_image_test(args.test_file_path)
        print(" [*] test finished")

if __name__ == '__main__':
    main()