def main():
    from aser.server import ASERServer
    from aser.utils.config import get_server_args_parser
    parser = get_server_args_parser()
    args = parser.parse_args()
    ASERServer(args)

if __name__ == "__main__":
    main()
