import kagglehub

def download_data(path):

    path = kagglehub.dataset_download(path)
    print("Path to dataset files:", path)

if __name__ == "__main__":
    download_data(path = r"redzapdos123/modified-aerial-traffic-and-visdrone-dataset-yolo")
