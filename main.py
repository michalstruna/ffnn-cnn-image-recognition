#!/usr/bin/python3

import init
import numpy as np
import os
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
        positive_dir = args.dir[0] if args.dir else os.path.join("img", "pos", "test")
        negative_dir = args.dir[1] if args.dir else os.path.join("img", "neg", "test")
        inputs, outputs = io_utils.prepare_input_set([positive_dir, negative_dir])

        network.load()
        trues, falses = network.test(inputs, outputs)
        console.print_test(network, trues, falses)

    elif args.train:
        positive_dir = args.dir[0] if args.dir else os.path.join("img", "pos", "train")
        negative_dir = args.dir[1] if args.dir else os.path.join("img", "neg", "train")
        inputs, outputs = io_utils.prepare_input_set([positive_dir, negative_dir], True)

        start = time.time()
        history = network.train(inputs, outputs, int(args.train[0]), int(args.train[1]), args.train[2])
        end = time.time()

        network.save()
        console.print_training(network, history, end - start)
    else:
        print("\n\n" + str(network) + "\n")
        network.load()
        network.summary()
        pass

console.show()
