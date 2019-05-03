from modules.const import *

from enum import Enum

class Command(Enum):
    TRAIN = 1
    EVAL = 2
    CLEAN = 3
    DIAGNOSE = 4
    FIND_LRATE = 5
    LABEL_IMAGES = 6

def call_command(args):
    import nn_network

    if args.command == Command.FIND_LRATE:        
        nn_network.find_best_learning_rate(args.input, args.steps)
    elif args.command == Command.TRAIN:
        nn_network.train_and_eval(args.input, args.steps)
    elif args.command == Command.EVAL:
        nn_network.evaluate(args.input)
    elif args.command == Command.CLEAN:
        nn_network.clean()
    elif args.command == Command.DIAGNOSE:        
        nn_network.show_wrong_predictions(args.input, args.subset)    
    elif args.command == Command.LABEL_IMAGES:
        nn_network.label_images(args.input, args.output, NUM_CORNERS)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Commands for training, evaluating and testing the neural network")    
    subparsers = parser.add_subparsers(help="Commands")
    subparsers.required = True

    train_parser = subparsers.add_parser('train', help="Train and evaluate the NN",
        description="Train and evaluate the NN")    
    train_parser.add_argument("-i","--input", type=str, default="dataset", help="Input dataset")    
    train_parser.add_argument("-s", "--steps", type=int, default=10**6, help="Number of learning steps for which to train the network")
    train_parser.set_defaults(command=Command.TRAIN)

    clean_parser = subparsers.add_parser('clean', help="Delete the model",
        description="Delete the model")    
    clean_parser.set_defaults(command=Command.TRAIN)

    lr_parser = subparsers.add_parser('lrate', help="Find the best learning rate",
        description="Find the best learning rate")    
    lr_parser.add_argument("-i","--input", type=str, default="dataset", help="Input dataset")    
    lr_parser.add_argument("-s", "--steps", type=int, default=1, help="Number of learning steps to make in this test")
    lr_parser.set_defaults(command=Command.FIND_LRATE)

    eval_parser = subparsers.add_parser('eval', help="Evaluate the NN",
        description="Evaluate the NN")    
    eval_parser.add_argument("-i","--input", type=str, default="dataset", help="Input dataset")
    eval_parser.add_argument("-l", "--labels", type=str, default="generated_labels.txt", help="Labels file")
    eval_parser.set_defaults(command=Command.EVAL)

    label_parser = subparsers.add_parser('label', help="Label data and save the labels to a file",
        description="Label data and save the labels to a file")    
    label_parser.add_argument("-i","--input", type=str, default="dataset", help="Input dataset")
    label_parser.add_argument("-o","--output", type=str, default="generated_labels.txt", help="Output file")    
    label_parser.set_defaults(command=Command.LABEL_IMAGES)
    
    show_parser = subparsers.add_parser('show', help="Show incorrect examples",
        description="Show incorrect examples")    
    show_parser.add_argument("-i","--input", type=str, default="dataset", help="Input dataset")    
    show_parser.add_argument("-s","--subset", type=str, default="test", help="Subfolder of the input folder. '.' if no subsets")
    show_parser.set_defaults(command=Command.DIAGNOSE)
    
    args = parser.parse_args()
    call_command(args)
