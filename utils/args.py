import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument("--name",
    #                     required=True,
    #                     help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["mnist"],
                        default="mnist",
                        help="Which dataset.")
    parser.add_argument('--data_root',
                        type=str,
                        default="E:\desktop\mnist",
                        help="where the dataset is")
    parser.add_argument("--model_type",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        default="resnet18",
                        help="Which variant to use.")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--epoch",
                        type=int,
                        default=30)
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3)
    parser.add_argument("--seed",
                        type=int,
                        default=100)
    parser.add_argument("--n_gpu",
                        type=int,
                        default=1)
    parser.add_argument("--save_model_path",
                        type=str,
                        default='./model')

    args = parser.parse_args()

    return args
