#!/usr/bin/python3

import init
import numpy as np
import time

import io_utils

console = io_utils.Console()
args = io_utils.read_args()
networks = io_utils.get_networks(args.use, 5)

for network in networks:
    if args.run:
        network.load()
        img = io_utils.read_img(args.run)
        result = network.run(np.array([img]))
        console.print_run(network, result)

    elif args.test:
        #positive_dir = args.dir[0] if args.dir else os.path.join("img", "pos", "test")
        #negative_dir = args.dir[1] if args.dir else os.path.join("img", "neg", "test")
        #inputs, outputs = io_utils.prepare_input_set([positive_dir, negative_dir])

        inputs, targets = io_utils.prepare_input_set("img/heads_and_non_heads/test", transform_input=network.transform_input)

        network.load()
        trues, falses = network.test(inputs, targets)
        console.print_test(network, trues, falses)

    elif args.train:
        #positive_dir = args.dir[0] if args.dir else os.path.join("img", "pos", "train")
        #negative_dir = args.dir[1] if args.dir else os.path.join("img", "neg", "train")
        #inputs, outputs = io_utils.prepare_input_set([positive_dir, negative_dir], True)

        inputs, targets = io_utils.prepare_input_set("img/heads_and_non_heads/train", train=True, transform_input=network.transform_input)

        start = time.time()
        history = network.train(inputs, targets, int(args.train[0]), int(args.train[1]), float(args.train[2]))
        end = time.time()

        network.save()
        console.print_training(network, history, end - start)
    else:
        print("\n\n" + str(network) + "\n")
        network.load()
        network.summary()
        pass

console.show()
