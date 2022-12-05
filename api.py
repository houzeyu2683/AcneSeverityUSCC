
import bucket
import network

def createCase(path):

    import PIL.Image

    Case = bucket.createClass(name='Case')
    case = Case()
    picture = PIL.Image.open(path).convert("RGB")
    picture = bucket.v1.transformPicture(picture=picture, inference=True)
    picture = picture.unsqueeze(0)
    case.picture = picture
    # createCase(path=api_config['demo']['higher'])
    return(case)

def downloadModel():

    import os 
    import gdown

    ##  Image classifier model.
    config = api_config['image-classifier']
    path = config['path']
    url = config['url']
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    gdown.download(url, path, quiet=False)    

    ##  Image classifier model.
    config = api_config['image-embedding']
    path = config['path']
    url = config['url']
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    gdown.download(url, path, quiet=False)    
    return

def loadModel():

    ##  Image classifier model.
    path = api_config['image-classifier']['path']
    image_classifier = network.v1.Machine(model=None)
    image_classifier.loadModel(path, device='cpu')
    image_classifier.model.eval()

    ##  Image classifier model.
    path = api_config['image-embedding']['path']
    image_embedding = network.v2.Machine(model=None)
    image_embedding.loadModel(path, device='cpu')
    image_embedding.model.eval()
    return(image_classifier, image_embedding)

def inferCase(case, models):
    
    import torch
    classifier, embedding = models
    
    with torch.no_grad():
        
        class_score = classifier.model.getScore(batch=case)
        class_prediction = class_score.argmax(1).item()
        class_extraction = classifier.model.getExtraction(batch=case)
        case.extraction = class_extraction
        pass

        class_attribute_prediction = embedding.model.getEncoding(batch=case)
        class_attribute_prediction = class_attribute_prediction.detach().numpy()[0,:].tolist()
        pass

    return(class_prediction, class_attribute_prediction)

##  U should load the model in the first, then start infer the case.
api_config = bucket.loadYaml(path='./api.yaml')
downloadModel()
models = loadModel()

case = createCase(path=api_config['demo']['higher'])
cls_pred, cls_attr_pred = inferCase(case, models)

case = createCase(path=api_config['demo']['lower'])
cls_pred, cls_attr_pred = inferCase(case, models)


