import argparse
import functools
import time

from mvector.trainer import MVectorTrainer
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/myres2net.yml')
add_arg("use_gpu",          bool,  True)
add_arg('save_image_path',  str,   'output/images/')
add_arg('resume_model',     str,   'models/CNCELEB/model.pth')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MVectorTrainer(configs=args.configs, use_gpu=args.use_gpu, overwrites=args.overwrites)

# 开始评估
start = time.time()
eer, min_dcf, threshold = trainer.evaluate(resume_model=args.resume_model, save_image_path=args.save_image_path)
end = time.time()
print('评估消耗时间：{}s，threshold：{:.5f}，EER: {:.5f}, MinDCF: {:.5f}'
      .format(int(end - start), threshold, eer, min_dcf))
