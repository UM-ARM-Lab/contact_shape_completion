#! /usr/bin/env python


import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools
from model.network import AutoEncoderWrapper
import IPython

shape_map = {"airplane":"02691156",
             "mug":"03797390"}


if __name__ == "__main__":
    print("Deprecated for now. Use `train.py`")
    return
    
    data_shapenet = data_tools.load_shapenet([shape_map["mug"]])



    # data = data_ycb
    data = data_shapenet
    data = data_tools.simulate_input(data, 10, 10, 10)


    params = {'num_latent_layers': 200}
    
    sn = AutoEncoderWrapper(params)
    # IPython.embed()

    sn.train_and_test(data)
    # sn.evaluate(data)
    # sn.restore()
    # sn.evaluate(data)
    # sn.evaluate(data)

