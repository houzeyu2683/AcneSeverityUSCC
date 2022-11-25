
import gdown

folder = {
    'name': 'v1',
    'url': 'https://drive.google.com/drive/folders/19rag2D_WDWrrXSBl6Qnk4c1rGHpefnaF'
}
gdown.download_folder(folder['url'], remaining_ok=True)
pass

folder = {
    'name': 'Classification',
    'url': 'https://drive.google.com/drive/folders/1rSmCbZyNyaiy5bkC-Eb2CQ4XxCB_RtcA'
}
# url = 'https://drive.google.com/drive/folders/1rSmCbZyNyaiy5bkC-Eb2CQ4XxCB_RtcA'
gdown.download_folder(folder['url'], remaining_ok=True)

folder = {
    'name': 'attribution',
    'url': 'https://drive.google.com/u/1/uc?id=1SQ1QONYKy3MKMsXwGnuc_9Aw8QdcA1Yy&export=download'
}
# url = "https://drive.google.com/u/1/uc?id=1SQ1QONYKy3MKMsXwGnuc_9Aw8QdcA1Yy&export=download"
gdown.download(folder["url"])

