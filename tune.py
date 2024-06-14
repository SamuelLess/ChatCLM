# Used as an adapter between wandb and rust code
import argparse
import json
import subprocess

import wandb
import os


def evaluate():
    run = wandb.init()

    params = ["accel", "d", "f", "k", "shrinkDict", "shrinkDictMaxRegression", "splitPoint", "steps", "compressionLevel"]

    # read all the arguments from wandb
    arguments = {param: getattr(wandb.config, param) for param in params}

    # Create a json object with the arguments
    arguments = json.dumps(dict(arguments))
    print("Starting with arguments:")
    print(arguments)
    # Start the binary, write the arguments to its stdin and read the output
    command = ["target/release/tuning"]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, stderr = process.communicate(arguments)
    print(output)
    print(stderr)



    # Parse the last line as json
    try:
        output = json.loads(output.splitlines()[-1])
        wandb.log(output)
    except json.JSONDecodeError:
        print("Error decoding json")
        wandb.log({})
    except IndexError:
        print("Error parsing output")
        wandb.log({})


# create sweep method: bayes
# metric:
#   goal: minimize
#   name: val_bpt
# parameters:
#   accel:
#     distribution: int_uniform
#     max: 10
#     min: 1
#   compressionLevel:
#     distribution: int_uniform
#     max: 18
#     min: 1
#   d:
#     values:
#       - 6
#       - 8
#   f:
#     distribution: int_uniform
#     max: 26
#     min: 5
#   k:
#     distribution: int_uniform
#     max: 2048
#     min: 16
#   shrinkDict:
#     values:
#       - 0
#       - 1
#   shrinkDictMaxRegression:
#     distribution: int_uniform
#     max: 50
#     min: 0
#   splitPoint:
#     distribution: uniform
#     max: 0.99
#     min: 0.5
#   steps:
#     distribution: int_uniform
#     max: 100
#     min: 0
sweep_config = {
    "method": "bayes",
    "metric": {
        "goal": "minimize",
        "name": "val_bpt"
    },
    "parameters": {
        "dataset_size": {
            "distribution": "int_uniform",
            "max": 1000000,
            "min": 100
        },
        "accel": {
            "distribution": "int_uniform",
            "max": 10,
            "min": 1
        },
        "compressionLevel": {
            "distribution": "int_uniform",
            "max": 18,
            "min": 1
        },
        "d": {
            "values": [6, 8]
        },
        "f": {
            "distribution": "int_uniform",
            "max": 26,
            "min": 5
        },
        "k": {
            "distribution": "int_uniform",
            "max": 2048,
            "min": 16
        },
        "shrinkDict": {
            "values": [0, 1]
        },
        "shrinkDictMaxRegression": {
            "distribution": "int_uniform",
            "max": 50,
            "min": 0
        },
        "splitPoint": {
            "distribution": "uniform",
            "max": 0.99,
            "min": 0.5
        },
        "steps": {
            "distribution": "int_uniform",
            "max": 100,
            "min": 0
        }
    }

}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id="nielsg/ChatCLM/lyzf1w0b", function=evaluate)





