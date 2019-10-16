def main():
    from aser.pipe import ASERPipe
    from aser.utils.config import get_pipe_args_parser
    parser = get_pipe_args_parser()
    args = parser.parse_args()
    ASERPipe(args).run()


