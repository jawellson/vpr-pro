'''
Author: jawellson 936575674@qq.com
Date: 2025-04-21 09:30:17
LastEditors: jawellson 936575674@qq.com
LastEditTime: 2025-05-09 17:08:42
FilePath: \vpr\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import argparse
import functools

from mvector.trainer import MVectorTrainer
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,    'configs/myres2net.yml')
add_arg('data_augment_configs', str,    'configs/augmentation.yml')
add_arg("local_rank",           int,    0)
add_arg("use_gpu",              bool,   True)
add_arg("do_eval",              bool,   True)
add_arg('save_model_path',      str,    'models/')
add_arg('log_dir',              str,    'log/')
add_arg('resume_model',         str,    None)
add_arg('pretrained_model',     str,    None)
add_arg('overwrites',           str,    None)
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MVectorTrainer(configs=args.configs,
                         use_gpu=args.use_gpu,
                         data_augment_configs=args.data_augment_configs,
                         overwrites=args.overwrites)

trainer.train(save_model_path=args.save_model_path,
              log_dir=args.log_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model,
              do_eval=args.do_eval)
