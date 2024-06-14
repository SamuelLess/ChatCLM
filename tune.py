# Used as an adapter between wandb and rust code
import argparse
import json
import subprocess

import wandb

sweep_config = {
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "train_inf_gain"
    },
    "parameters": {
        "datasetSize": {
            "distribution": "int_uniform",
            "max": 180_000_000,
            "min": 1_000_000,
        },
        "dictionarySizePercentage" : {
            "distribution": "uniform",
            "max": 1,
            "min": 0
        },

        "compressionLevel": {
            "distribution": "int_uniform",
            "max": 8,
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
    }

}

def evaluate():
    run = wandb.init()

    conf_dict = {}
    for param in sweep_config["parameters"].keys():
        conf_dict[param] = wandb.config[param]

    # Create a json object with the arguments
    arguments = json.dumps(conf_dict)
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




# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id=sweep_id, function=evaluate)





