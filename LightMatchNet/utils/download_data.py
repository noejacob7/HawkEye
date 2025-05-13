import kagglehub

veri_dataset = "abhyudaya12/veri-vehicle-re-identification-dataset"
lfw_dataset = "atulanandjha/lfwpeople"

# Download latest version
path = kagglehub.dataset_download(lfw_dataset)

print("Path to dataset files:", path)