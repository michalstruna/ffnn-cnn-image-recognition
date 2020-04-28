#!/usr/bin/python3

from io_utils import read_args, get_networks

args = read_args()
networks = get_networks(args.use)

for network in networks:
    print("=======", network, "=======")