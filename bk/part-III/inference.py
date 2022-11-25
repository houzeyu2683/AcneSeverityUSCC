
import data
import network

import torch.nn

if(__name__=='__main__'):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default='./Sample/levle3_88.jpg', help="image path", type=str)
    parser.add_argument("--model", default='./output/checkpoint-0/acne-classifier.pt', help="model path", type=str)
    args = parser.parse_args()
    pass

    interface = data.Interface(path=args.image)
    case = interface.createCase()
    pass

    machine = network.v1.machine()
    machine.loadModel(args.model, device='cpu')
    score = machine.model.getScore(case).detach()
    softmax = torch.nn.Softmax(dim=1)
    score = softmax(score).numpy().flatten().round(3).tolist()
    print("Severity score: {}".format(score))
    pass
