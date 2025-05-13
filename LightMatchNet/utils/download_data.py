import kagglehub

veri_dataset = "abhyudaya12/veri-vehicle-re-identification-dataset"
lfw_dataset = "atulanandjha/lfwpeople"

# Download latest version
path = kagglehub.dataset_download()

print("Path to dataset files:", path)