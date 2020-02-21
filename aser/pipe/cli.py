import os
from aser.pipe import ASERPipe
from aser.utils.config import get_pipe_args_parser

def main():
    parser = get_pipe_args_parser()
    args = parser.parse_args()
    if args.raw_dir.endswith(os.sep):
        args.raw_dir = args.raw_dir[:-1]
    if args.processed_dir.endswith(os.sep):
        args.processed_dir = args.processed_dir[:-1]
    if args.full_kg_dir.endswith(os.sep):
        args.full_kg_dir = args.full_kg_dir[:-1]
    if args.core_kg_dir.endswith(os.sep):
        args.core_kg_dir = args.core_kg_dir[:-1]
    ASERPipe(args).run()

if __name__ == "__main__":
    main()