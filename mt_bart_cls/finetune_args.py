import argparse


parser = argparse.ArgumentParser()

# 模型读取与保存
parser.add_argument("--model_name_or_path", default="facebook/bart-large", type=str, required=True, help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.")
parser.add_argument("--output_dir", required=True, type=str, help="The model checkpoint saving path")

# 训练模型用到的数据文件路径
parser.add_argument("--data_dir", type=str, default="", help='the path of the dataset')
parser.add_argument("--log_file", type=str, default="", help='train log file')
parser.add_argument("--dialogue_name_prefix", type=str, default="simmc2.1_dials_dstc11_", help='dialogue_name_prefix of the json file')
parser.add_argument("--jsons_dir_name", type=str, default="simmc2_scene_jsons_dstc10_public", help='jsons_dir_name')
parser.add_argument("--images_dir_name", type=str, default="simmc2_scene_images_dstc10_public", help='images_dir_name')
parser.add_argument("--train_input_file", required=True, type=str, help='preprocessed input file path')
parser.add_argument("--disambiguation_file", type=str, help='preprocessed input file path')
parser.add_argument("--response_file", type=str, help='preprocessed input file path, line-by-line format')
parser.add_argument("--train_target_file", type=str, help='preprocessed target file path, line-by-line format')
parser.add_argument("--eval_input_file", type=str, help='preprocessed input file path, line-by-line format')
parser.add_argument("--eval_target_file", type=str, help='preprocessed target file path, line-by-line format')
parser.add_argument("--add_special_tokens", default=None, required=True, type=str, help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.")
parser.add_argument("--item2id", required=True, type=str, help='item2id filepath')

# 模型训练时的超参数
parser.add_argument('--num_train_epochs', default=3, type=int, help="train epochs")
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument('--train_batch_size', default=4, type=int)
parser.add_argument('--eval_batch_size', default=4, type=int)
parser.add_argument('--logging_steps', default=10, type=int, help="logging steps")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps(warm_up_ratio*t_total).")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--embedding_train_steps", default=200, type=int)
parser.add_argument("--embedding_train_epochs_start", type=int, default=400)
parser.add_argument("--embedding_train_epochs_ongoing", type=int, default=100)
parser.add_argument("--do_train_embedding_clip_way_during_training", action="store_true", help="Run train_embedding_clip_way during training at each embedding_train_step.")
parser.add_argument("--do_retrieval", action="store_true", help="do_retrieval during training.")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--fp16", action="store_true", help="fp16 training.")

parser.add_argument("--lambda_lm_loss", default=1.0, type=float)
parser.add_argument("--lambda_nocoref_loss", default=0.1, type=float)
parser.add_argument("--lambda_misc_loss", default=0.1, type=float)
parser.add_argument("--lambda_disam_loss", default=0.1, type=float)
parser.add_argument("--lambda_dst_loss", default=1.0, type=float)


# 多卡训练配置
parser.add_argument('--per_gpu_train_batch_size', default=4, type=int)
parser.add_argument('--n_gpu', default=4, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
# 模型切分训练
parser.add_argument("--model_parallel", action="store_true", help="model parallel")

parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_steps", default=2000, type=int)
parser.add_argument("--output_eval_file", type=str, default=".txt file")
