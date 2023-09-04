# Constants
FLAIR1_NUM_CLASSES = 13

CLASS_LABELS = [
    "building",
    "pervious surface",
    "impervious surface",
    "bare soil",
    "water",
    "coniferous",
    "deciduous",
    "brushwood",
    "vineyard",
    "herbaceous vegetation",
    "agricultural land",
    "plowed land",
    "other",
]

OCSGE_LUT = [
 (219,  14, 154),
 (114, 113, 112),
 (248,  12,   0),
 ( 61, 230, 235),
 (169, 113,   1),
 ( 21,  83, 174),
 (255, 255, 255),
 (138, 179, 160),
 ( 70, 228, 131),
 ( 25,  74,  38),
 (243, 166,  13),
 (102,   0, 130),
 (255, 243,  13),
 (228, 223, 124),
  (128, 0, 255),
 (64, 128, 128),
 (223, 0, 223),
 (128, 128, 192),
 (  0,   0,   0),
]

LUT_COLORS = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

LUT_CLASSES = {
1   : 'building',
2   : 'pervious surface',
3   : 'impervious surface',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous vegetation',
11  : 'agricultural land',
12  : 'plowed land',
13  : 'swimming_pool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}

ALAN_CLASSES = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']

ALAN_ID2LABEL = {
    0: 'None',
    1: 'building',
    2: 'pervious surface',
    3: 'impervious surface',
    4: 'bare soil',
    5: 'water',
    6: 'coniferous',
    7: 'deciduous',
    8: 'brushwood',
    9: 'vineyard',
    10: 'herbaceous vegetation',
    11: 'agricultural land',
    12: 'plowed land'}
