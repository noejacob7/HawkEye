import xml.etree.ElementTree as ET

def parse_veri_labels(xml_path):
    """
    Parses the VeRi label XML file with gb2312 encoding
    Returns: imageName -> (vehicleID, colorID, typeID)
    """
    id_to_info = {}

    with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
        xml_content = f.read()

    root = ET.fromstring(xml_content)

    for item in root.findall(".//Item"):
        image = item.attrib["imageName"]
        vid = item.attrib["vehicleID"]
        color = item.attrib["colorID"]
        vtype = item.attrib["typeID"]
        id_to_info[image] = (vid, color, vtype)

    return id_to_info
