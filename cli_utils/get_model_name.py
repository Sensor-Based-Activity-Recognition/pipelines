import os
import yaml

if __name__ == "__main__":
    # get model name from params.yaml
    with open(os.path.join(os.path.dirname(__file__), "../params.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    modelname = params["dvclive"]["model"]
    print(modelname)
