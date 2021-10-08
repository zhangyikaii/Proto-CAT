from models.utils import (
    get_command_line_parser, 
    preprocess_args,
    pprint
)
from models.train import (
    Trainer
)

if __name__ == '__main__':
    """
    准备命令行参数
    """
    parser = get_command_line_parser()
    args = preprocess_args(parser.parse_args())
    pprint(vars(args))

    try:
        """
        Training
        """
        trainer = Trainer(args)
        trainer.fit()
        trainer.test()
    except KeyboardInterrupt:
        trainer.delete_logs()