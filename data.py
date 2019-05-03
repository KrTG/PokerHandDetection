from modules.const import *
from modules import card_detection
from modules import interpret_labels

import cv2

import os  
import glob
import shutil
import random
import multiprocessing  
import functools
import json

def copy_example(path_mapping):    
    print(path_mapping[1])
    shutil.copy(path_mapping[0], path_mapping[1])

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def copy_folder_structure(input_folder, output_folder):
    if not os.path.exists(output_folder):
        shutil.copytree(input_folder, output_folder, ignore=ignore_files)
    else:
        print("Folder: '{}' already exists".format(output_folder))    

def create_labeled_folders(folder):      
    for i in range(NUM_CLASSES):
        labeled_folder = os.path.join(folder, str(i))
        try:
            os.mkdir(labeled_folder)
        except FileExistsError:
            print("Folder: '{}' already exists".format(labeled_folder))                

def get_leafs(paths):
    result = []
    for path in paths:
        if not any(x.startswith(path) for x in paths if x != path):
            result.append(path)
    return set(result)

def extract_corners(input_folder, output_folder, num_corners):         
    copy_folder_structure(input_folder, output_folder)        
    paths = glob.glob(os.path.join(input_folder, "**", "*.jpg"), recursive=True)
            
    card_index = 0
    for path in paths:
        image = cv2.imread(path)        
        cards = card_detection.detect_cards(image)

        for card in cards:
            extracted_corners = card_detection.cut_corners(image, card, num_corners)           
            for corner_index, corner_image in enumerate(extracted_corners):                                
                folder = os.path.split(os.path.relpath(path, start=input_folder))[0]
                pathname = os.path.join(output_folder, folder, "{}-{}.png".format(card_index, corner_index))
                print(pathname)
                cv2.imwrite(pathname, corner_image)
            card_index += 1

def split_by_label(input_folder, output_folder, label_file):
    copy_folder_structure(input_folder, output_folder)
    subfolders = get_leafs(glob.glob(os.path.join(output_folder, "**/"), recursive=True))    
    for subfolder in subfolders:
        create_labeled_folders(subfolder)

    label_dict = {}
    with open(label_file, 'r') as fp:
        label_list = json.load(fp) 
        for entry in label_list:
            try:
                label = interpret_labels.get_label(entry[1], entry[2])                
                label_dict[entry[0]] = label
            except KeyError:
                print("Example {} has an invalid label".format(entry[0]))
    
    paths = glob.glob(os.path.join(input_folder, "**", "*.png"), recursive=True)

    path_mapping = []
    for path in paths:
        basename = os.path.basename(path)
        index = int(basename.split("-")[0]) 
        try:
            label = label_dict[index]
            folder = os.path.split(os.path.relpath(path, start=input_folder))[0]
            new_path = os.path.join(output_folder, folder, str(label), basename)                         
            path_mapping.append((path, new_path))
        except KeyError:
            pass

    with multiprocessing.Pool(NUM_THREADS) as p:
        p.map(copy_example, path_mapping)


def resize_image(path_mapping, new_size):
    image = cv2.imread(path_mapping[0])
    resized = cv2.resize(image, (new_size, new_size))
    cv2.imwrite(path_mapping[1], resized)
    print(path_mapping[1])

def resize_images(input_folder, output_folder, new_size):
    copy_folder_structure(input_folder, output_folder)

    paths = glob.glob(os.path.join(input_folder, "**", "*.png"), recursive=True)

    path_mapping = []
    for path in paths:
        relpath = os.path.relpath(path, start=input_folder)                 
        new_path = os.path.join(output_folder, relpath)                         
        path_mapping.append((path, new_path))

    with multiprocessing.Pool(NUM_THREADS) as p:
        p.map(functools.partial(resize_image, new_size=new_size), path_mapping)

def extract_labels(input_folder, output_file):
    paths = glob.glob(os.path.join(input_folder, "**", "*.png"), recursive=True)
    paths = sorted(paths, key=lambda x: int(os.path.basename(x).split("-")[0]))

    label_list = []        
    for path in paths[::NUM_CORNERS]:
        labeled_folder, basename = os.path.split(path)
        label = int(os.path.basename(labeled_folder))
        index = int(basename.split("-")[0])
        color, rank = interpret_labels.get_classes(label)
        label_list.append([index, color, rank])

    with open(output_file, "w") as f:
        json.dump(label_list, f)        
        
def split_dataset(input_folder, output_folder, ratio):
    try:                
        os.mkdir(output_folder)        
    except FileExistsError:
        print("Folder: '{}' already exists".format(output_folder))
    copy_folder_structure(input_folder, os.path.join(output_folder, "train"))
    copy_folder_structure(input_folder, os.path.join(output_folder, "test"))

    paths = glob.glob(os.path.join(input_folder, "**", "*.png"), recursive=True)
    random.shuffle(paths)        

    path_mapping = []
    for index, path in enumerate(paths):
        subset = "train" if index / len(paths) < ratio else "test"
        relpath = os.path.relpath(path, start=input_folder)        
        new_path = os.path.join(output_folder, subset, relpath)
        path_mapping.append((path, new_path))

    with multiprocessing.Pool(NUM_THREADS) as p:
        p.map(copy_example, path_mapping)

def balance_dataset(input_folder, output_folder):
    copy_folder_structure(input_folder, output_folder)
    
    all_folders = glob.glob(os.path.join(output_folder, "**/"), recursive=True)
    parent_folders = [os.path.split(os.path.split(x)[0])[0] for x in all_folders]
    subfolders = get_leafs(parent_folders)    

    for subfolder in subfolders:        
        input_subfolder = os.path.join(input_folder, os.path.relpath(subfolder, start=output_folder))                
        output_subfolder = subfolder

        files_to_save = []        
        #Find the paths to files for each class in each subset        
        label_files = { label: glob.glob(os.path.join(input_subfolder, str(label), "*.png"), recursive=True) for label in range(NUM_CLASSES) }
        min_size = min(len(label_files[label]) for label in range(NUM_CLASSES))
        
        #shuffle the files
        for file_array in label_files.values():
            random.shuffle(file_array)

        for label in range(NUM_CLASSES):
            files_to_save += label_files[label][:min_size]

        path_mapping = []
        for path in files_to_save:
            relpath = os.path.relpath(path, start=input_subfolder)        
            new_path = os.path.join(output_subfolder, relpath)
            path_mapping.append((path, new_path))
        
        with multiprocessing.Pool(NUM_THREADS) as p:
            p.map(copy_example, path_mapping)        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract data for training")
    subparsers = parser.add_subparsers(help="Operation to be performed", dest="command")
    subparsers.required = True

    corner_parser = subparsers.add_parser("corners", help="Extract the card corners from the image",
        description="Extract the card corners from the image")
    corner_parser.add_argument("-i", "--input", type=str, default="dataset", help="Input data folder (Images with varying amounts of cards)")
    corner_parser.add_argument("-o", "--output", type=str, default="corners", help="Output data folder")        
    corner_parser.add_argument("-c", "--corners", type=int, default=4, help="Number of corners")
    corner_parser.set_defaults(func=lambda args: extract_corners(input_folder=args.input, output_folder=args.output, num_corners=args.corners))

    #### SPLIT OPS

    split_parser = subparsers.add_parser("split", help="Split the images into subfolders",
        description="Split the images into subfolders")
    split_subparsers = split_parser.add_subparsers(help="How to split the data", dest="criterium")
    split_subparsers.required = True

    split_by_label_parser = split_subparsers.add_parser("label", help="Split the data into labeled folders",
        description="Split the data into labeled folders")
    split_by_label_parser.add_argument("-i", "--input", type=str, default="resized", help="Input data folder (Images with varying amounts of cards)")
    split_by_label_parser.add_argument("-o", "--output", type=str, default="labeled", help="Output data folder")
    split_by_label_parser.add_argument("-l", "--labels", type=str, default="labels.txt", help="File with labels")    
    split_by_label_parser.set_defaults(func=lambda args: split_by_label(input_folder=args.input, output_folder=args.output, label_file=args.labels))

    split_by_subset_parser = split_subparsers.add_parser("subset", help="Split the dataset into train and test set",
        description="Split the dataset into train and test set")
    split_by_subset_parser.add_argument("-i", "--input", type=str, default="resized", help="Input data folder")
    split_by_subset_parser.add_argument("-o", "--output", type=str, default="split", help="Output data folder")    
    split_by_subset_parser.add_argument("-r", "--ratio", type=float, default=0.9, help="Ratio")
    split_by_subset_parser.set_defaults(func=lambda args: split_dataset(input_folder=args.input, output_folder=args.output, ratio=args.ratio))

    ####

    resize_parser = subparsers.add_parser("resize", help="Resize the images to the given size",
        description="Resize the images to the given size")
    resize_parser.add_argument("-i", "--input", type=str, default="cropped", help="Input data folder")
    resize_parser.add_argument("-o", "--output", type=str, default="resized", help="Output data folder")
    resize_parser.add_argument("-s", "--size", type=int, default=ISIZE, help="New image size (width and height)")
    resize_parser.set_defaults(func=lambda args: resize_images(input_folder=args.input, output_folder=args.output, new_size=args.size))

    label_parser = subparsers.add_parser("labelfile", help="Create a label file from labeled folders",
        description="Create a label file")
    label_parser.add_argument("-i", "--input", type=str, default="dataset", help="Input data folder")
    label_parser.add_argument("-o", "--output", type=str, default="labels.txt", help="Output file with labels")
    label_parser.set_defaults(func=lambda args: extract_labels(input_folder=args.input, output_file=args.output))

    balance_parser = subparsers.add_parser("balance", help="Balance the classes for the given dataset",
        description="Balance the classes for the given dataset")
    balance_parser.add_argument("-i", "--input", type=str, help="input dataset")
    balance_parser.add_argument("-o", "--output", type=str, help="output dataset")    
    balance_parser.set_defaults(func=lambda args: balance_dataset(args.input, args.output))

    args = parser.parse_args()
    args.func(args)
    