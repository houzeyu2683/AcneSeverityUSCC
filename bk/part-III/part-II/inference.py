
import data
import network
import torch
import torch.nn
import pickle
import os
import pandas
def getFeature(path):

    with open(path, 'rb') as paper:

        feature = pickle.load(paper)
        pass

    return(feature)

def createCase(feature):

    class case: pass
    case.feature = feature
    return(case)

if(__name__=='__main__'):

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default='./Sample/levle3_88.jpg', help="image path", type=str)
    # parser.add_argument("--model_path", default='./output/checkpoint-0/acne-classifier.pt', help="model path", type=str)
    # args = parser.parse_args()
    test_feature_path = './resource/ACNE04/Attribution/test-feature.pkl'
    image_path = './Sample/levle3_88.jpg'
    model_path = './output/checkpoint-19/acne-vae-weight.pt'
    pass

    model = network.v2.Model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # image_folder = './resource/ACNE04/Classification/JPEGImages/'
    ce_pred_table = []
    storage = 'output/ce_pred/'
    os.makedirs(storage, exist_ok=True)
    test_feature = getFeature(path=test_feature_path)
    # group = []
    for image_name in test_feature['index']:

        index = test_feature['index'].index(image_name)
        image_feature = test_feature['feature'][index:index+1,:]
        image_feature = torch.tensor(image_feature)
        case = createCase(feature=image_feature)
        value = model.forward(case)

        ce_pred = value['encode_feature'].detach().squeeze().numpy().round(3)
        print(ce_pred)
        print("這裡是針對給定的影像的 2048 特徵進行 encode 產生 CE 結果。")
        ce_pred_row = pandas.DataFrame(ce_pred).transpose()

        item = pandas.DataFrame({"image":[image_name]})
        row = pandas.concat([item, ce_pred_row], axis=1)
        ce_pred_table += [row]
        continue        

    ce_pred_table = pandas.concat(ce_pred_table)        
    ce_pred_table.shape
    ce_pred_table.to_csv(os.path.join(storage, 'ce_pred_table.csv'))


    