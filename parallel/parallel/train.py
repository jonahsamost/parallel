from parallel.parallel.state import ParallelConfig, init_dist
from parallel.parallel.utils import load_cfg
import transformers


def main():
    cfg = load_cfg()
    init_dist(cfg)
    pconfig = ParallelConfig(cfg)


if __name__ == '__main__':
    main()