from evaluator import COCOPanopticEvaluatorMod
from train_net import Trainer, DetectionCheckpointer, setup, launch, default_argument_parser
from detectron2.evaluation import DatasetEvaluator

class SaveResultsEvaluator(DatasetEvaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def process(self, inputs, outputs):
        print(inputs, outputs)


def main(args):
    cfg = setup(args, use_wandb=False)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    res = Trainer(cfg).test(cfg, model, force=True)



if __name__ == "__main__":
    import sys
    sys.argv[1:] = '--config-file EMSANet/external/configs/NYUv2/hm3d.yaml DATASETS.BASE_PATH /home/sai/Desktop/multi-object-search/EMSANet/datasets/hm3d MODEL.WEIGHTS /home/sai/Desktop/multi-object-search/EMSANet/external/results/model_final.pth'.split(' ')

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
