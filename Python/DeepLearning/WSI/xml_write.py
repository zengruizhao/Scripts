# coding=utf-8
import os
import scipy.io as sio
from time import time
from lxml import etree
from collections import OrderedDict


class XMLWritter:
    def __init__(self, data_path, save_path, skew, skip_record=5):
    # skip_record:隔几个点存一个坐标
        self.data_path = data_path
        self.save_path = save_path
        self.data_list = []
        self.skew_pixles_40x = skew
        self.skip_record = skip_record
        self.default_color_dict = OrderedDict([("r", "0000ff"), ("g", "00ff00"), ("b", "ff0000"),
                                               ("black", "000000"), ("pink", "ffC0cb"), ("purple", "800080"),
                                               ("c", "00ffff"), ("gray", "808080"), ("sky", "87cecb"),
                                               ("gold", "ffd700"), ("brown", "a52a2a"), ("tomato", "ff6347")])
        self.annotation_attribute_name_dict = OrderedDict([("Id", "1"), ("Name", ""), ("ReadOnly", "0"),
                                                           ("NameReadOnly", "0"), ("LineColorReadOnly", "0"),
                                                           ("Incremental", "0"), ("Type", "4"), ("LineColor", "65280"),
                                                           ("Visible", "1"), ("Selected", "1"), ("MarkupImagePath", ""),
                                                           ("MacroName", "")])
        self.region_attribute_name_dict = OrderedDict([("Id", "1"), ("Type", "0"), ("Zoom", "0.0"), ("Selected", "0"),
                                                      ("ImageLocation", ""), ("ImageFocus", "-1"), ("Length", "0.1"),
                                                      ("Area", "0.1"), ("LengthMicrons", "0.1"), ("AreaMicrons", "0"),
                                                      ("Text", ""), ("NegativeROA", "0"), ("InputRegionId", "0"),
                                                      ("Analyze", "1"), ("DisplayId", "1")])
        self.get_data_list()

    def get_data_list(self):
        import os
        for root, _, fnames in os.walk(self.data_path):
            for fname in fnames:
                if fname.endswith('.mat'):
                    self.data_list.append(os.path.join(root, fname))

    def create_annotation_attribute(self, node, id='1', name='', read_only='0',
                                    name_read_only='0', line_color_read_only='0',
                                    incremental='0', type='4', line_color='65280',
                                    visible='1', selected='1', markup_image_path='', macro_name=''):
        self.annotation_attribute_name_dict["Id"] = id
        self.annotation_attribute_name_dict["Name"] = name
        self.annotation_attribute_name_dict["ReadOnly"] = read_only
        self.annotation_attribute_name_dict["NameReadOnly"] = name_read_only
        self.annotation_attribute_name_dict["LineColorReadOnly"] = line_color_read_only
        self.annotation_attribute_name_dict["Incremental"] = incremental
        self.annotation_attribute_name_dict["Type"] = type
        self.annotation_attribute_name_dict["LineColor"] = line_color
        self.annotation_attribute_name_dict["Visible"] = visible
        self.annotation_attribute_name_dict["Selected"] = selected
        self.annotation_attribute_name_dict["MarkupImagePath"] = markup_image_path
        self.annotation_attribute_name_dict["MacroName"] = macro_name
        for key in self.annotation_attribute_name_dict:
            node.set(key, self.annotation_attribute_name_dict[key])
        return node

    def create_region_attribute(self, node, id='1', type='0', zoom="0.0", selected="0",
                                image_location="", image_focus="-1", length="0.1", area="0.1",
                                length_microns="0.1", area_microns="0", text="", negative_roa="0",
                                input_region_id="0", analyze="1", display_id="1"):
        self.region_attribute_name_dict["Id"] = id
        self.region_attribute_name_dict["Type"] = type
        self.region_attribute_name_dict["Selected"] = selected
        self.region_attribute_name_dict["ImageLocation"] = image_location
        self.region_attribute_name_dict["ImageFocus"] = image_focus
        self.region_attribute_name_dict["Length"] = length
        self.region_attribute_name_dict["Area"] = area
        self.region_attribute_name_dict["Type"] = type
        self.region_attribute_name_dict["LengthMicrons"] = length_microns
        self.region_attribute_name_dict["AreaMicrons"] = area_microns
        self.region_attribute_name_dict["Selected"] = selected
        self.region_attribute_name_dict["Text"] = text
        self.region_attribute_name_dict["NegativeROA"] = negative_roa
        self.region_attribute_name_dict["InputRegionId"] = input_region_id
        self.region_attribute_name_dict["Analyze"] = analyze
        self.region_attribute_name_dict["DisplayId"] = display_id
        for key in self.region_attribute_name_dict:
            node.set(key, self.region_attribute_name_dict[key])
        return node

    def create_attribute_header(self, node):
        RegionAttributeHeaders = node
        key = ['Id', 'Name', 'ColumnWidth']
        id_value = ['9999', '9997', '9996', '9998']
        name_value = ['Region', 'Length', 'Area', 'Text']
        column_width_value = ['-1'] * len(id_value)
        for one_p in zip(id_value, name_value, column_width_value):
            node = etree.SubElement(RegionAttributeHeaders, 'AttributeHeader')
            for ind, one_k in enumerate(key):
                node.set(one_k, one_p[ind])

    def create_vertex(self, node, data):
        """
        :param node: parent node
        :param data: X, Y, Z=0, X->width, Y->height
        :return: node after created
        """
        node = etree.SubElement(node, 'Vertex')
        node.set('X', str(data[1] + self.skew_pixles_40x))
        node.set('Y', str(data[0] + self.skew_pixles_40x))
        node.set('Z', '0')
        return node

    def write_vertices(self, node, data):
        Regions = node
        for ind, one_region in enumerate(data):
            Region = etree.SubElement(Regions, 'Region')
            self.create_region_attribute(Region, id=str(ind + 1))
            etree.SubElement(Region, 'Attributes')
            node = etree.SubElement(Region, 'Vertices')
            vertex = one_region[0]
            for i, one_vertex in enumerate(vertex):
                if (i + self.skip_record / 2 + 1) % self.skip_record == 0:
                    self.create_vertex(node, one_vertex)
            self.create_vertex(node, vertex[2])

    def main_writter(self):
        for index, one in enumerate(self.data_list):
            st_t = time()
            name = one.split('/')[-1].split('_')[0]
            data = sio.loadmat(one)['Region'][0]
            #################################################
            Annotations = etree.Element('Annotations')
            Annotations.set("MicronsPerPixel", "0.441600")

            Annotation = etree.SubElement(Annotations, 'Annotation')
            Annotation = self.create_annotation_attribute(Annotation)

            etree.SubElement(Annotation, 'Attributes')

            Regions = etree.SubElement(Annotation, 'Regions')

            RegionAttributeHeaders = etree.SubElement(Regions, 'RegionAttributeHeaders')
            self.create_attribute_header(RegionAttributeHeaders)
            self.write_vertices(Regions, data)

            etree.SubElement(Annotation, 'Plots')
            #################################################

            tree = etree.ElementTree(Annotations)
            tree.write(os.path.join(self.save_path, name + '.xml'), pretty_print=True)
            print('Finished {}, {}/{}, time: {}s'.format(name+'.xml', (index+1), len(self.data_list), (time()-st_t)))


def main():
    data_path = '/mnt/Darwin/Cholangiocarcinoma/map/boundary_mat/'
    save_path = '/mnt/Darwin/Cholangiocarcinoma/map/xml_show/'

    writter = XMLWritter(data_path, save_path, skew=200)
    writter.main_writter()


if __name__ == "__main__":
    main()

