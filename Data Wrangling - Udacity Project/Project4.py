import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import csv
import cerberus
import schema
import codecs


OSMFILE = "map"
SAMPLE_FILE = "sample.osm"

k = 10

type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road",
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "PLACE": "Place",
            "Ave" : "Avenue",
            "in" : "IN",
            "Indiana" : "IN",
            "46410-5468" : "46410",
            "portage" : "Portage",
            }


def audit(osmfile):
    osm_file = open(osmfile, "r")
    types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                elif is_state_name(tag):
                    audit_state_name(state_types, tag.attrib['v'])
                elif is_postcode_number(tag):
                    audit_postcode_number(postcode_types, tag.attrib['v'])
                elif is_city_name(tag):
                    audit_city_name(city_types, tag.attrib['v'])
    osm_file.close()
    return types

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit_street_type(street_types, street_name):
    m = type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def is_state_name(elem):
    return (elem.attrib['k'] == "addr:state")

def audit_state_type(state_types, state_name):
    m = type_re.search(state_name)
    if m:
        state_type = m.group()
        if state_type not in expected:
            state_types[state_type].add(state_name)

def is_postcode_number(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_postcode_type(postcode_types, postcode_number):
    m = type_re.search(postcode_number)
    if m:
        postcode_type = m.group()
        if postcode_type not in expected:
            postcode_types[postcode_type].add(postcode_number)

def is_city_name(elem):
    return (elem.attrib['k'] == "addr:city")

def audit_city_type(state_types, state_name):
    m = type_re.search(city_name)
    if m:
        city_type = m.group()
        if city_type not in expected:
            city_types[city_type].add(city_name)

def update_name(name, mapping):
    m = type_re.search(name)
    if m.group() not in expected:
        if m.group() in mapping.keys():
            name = re.sub(m.group(), mapping[m.group()], name)
    return name

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    # YOUR CODE HERE
    if element.tag == 'node':
        for attr in node_attr_fields:
            node_attribs[attr] = element.attrib[attr]
        for tag in element.iter("tag"):
            node_tags_attribs = {}
            if problem_chars.search(tag.attrib['k']):
                continue
            elif LOWER_COLON.search(tag.attrib['k'].lower()):
                node_tags_attribs['id'] = element.attrib['id']
                node_tags_attribs['key'] = tag.attrib['k'].split(":",1)[1]
                node_tags_attribs['value'] = tag.attrib['v']
                node_tags_attribs['type'] = tag.attrib['k'].split(":",1)[0]
                if tag.attrib['k'] == 'addr:street' or tag.attrib['k'] == 'addr:state' or tag.attrib['k'] == 'addr:postcode' or tag.attrib['k'] == 'addr:city':
                    node_tags_attribs['value'] = update_name(tag.attrib['v'], mapping)
                else:
                    node_tags_attribs['value'] = tag.attrib['v']
            else:
                node_tags_attribs['id'] = element.attrib['id']
                node_tags_attribs['key'] = tag.attrib['k']
                node_tags_attribs['value'] = tag.attrib['v']
                node_tags_attribs['type'] = default_tag_type
                if tag.attrib['k'] == 'addr:street' or tag.attrib['k'] == 'addr:state' or tag.attrib['k'] == 'addr:postcode' or tag.attrib['k'] == 'addr:city':
                    node_tags_attribs['value'] = update_name(tag.attrib['v'], mapping)
                else:
                    node_tags_attribs['value'] = tag.attrib['v']
            tags.append(node_tags_attribs)
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        for attr in way_attr_fields:
             way_attribs[attr] = element.attrib[attr]
             position = 0
        for wn in element.iter('nd'):
            way_node_attribs ={}
            way_node_attribs['id'] = element.attrib['id']
            way_node_attribs['node_id'] = wn.attrib['ref']
            way_node_attribs['position'] = position
            position = position + 1
            way_nodes.append(way_node_attribs)
        for tag in element.iter('tag'):
            way_tag_attribs = {}
            if problem_chars.search(tag.attrib['k']):
                continue
            elif LOWER_COLON.search(tag.attrib['k'].lower()):
                way_tag_attribs['id'] = element.attrib['id']
                way_tag_attribs['key'] = tag.attrib['k'].split(":",1)[1]
                way_tag_attribs['value'] = tag.attrib['v']
                way_tag_attribs['type'] = tag.attrib['k'].split(":",1)[0]
                if tag.attrib['k'] == 'addr:street' or tag.attrib['k'] == 'addr:state' or tag.attrib['k'] == 'addr:postcode' or tag.attrib['k'] == 'addr:city':
                    way_tag_attribs['value'] = update_name(tag.attrib['v'], mapping)
                else:
                    way_tag_attribs['value'] = tag.attrib['v']
            else:
                way_tag_attribs['id'] = element.attrib['id']
                way_tag_attribs['key'] = tag.attrib['k']
                way_tag_attribs['value'] =  tag.attrib['v']
                way_tag_attribs['type'] = default_tag_type
                if tag.attrib['k'] == 'addr:street' or tag.attrib['k'] == 'addr:state' or tag.attrib['k'] == 'addr:postcode' or tag.attrib['k'] == 'addr:city':
                    way_tag_attribs['value'] = update_name(tag.attrib['v'], mapping)
                else:
                    way_tag_attribs['value'] = tag.attrib['v']
            tags.append(way_tag_attribs)
    return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}



# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSMFILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write('</osm>')


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)

        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSMFILE, validate=True)
