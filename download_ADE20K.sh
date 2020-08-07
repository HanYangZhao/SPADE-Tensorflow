    #!/bin/bash 
    curl -L -o ADEChallengeData2016.zip  https://github.com/HanYangZhao/SPADE-Tensorflow/releases/download/aed20k/ADEChallengeData2016.zip
    unzip ADEChallengeData2016.zip

    mkdir -p dataset/AED20K/image
    cp ADEChallengeData2016/images/training/* dataset/AED20K/image/


    mkdir -p dataset/AED20K/segmap
    cp ADEChallengeData2016/annotations/training/* dataset/AED20K/segmap/

    mkdir -p dataset/AED20K/segmap_test
    #copy random 5-10 files from the validation dataset to validation test set
    cp  ADEChallengeData2016/annotations/validation/ADE_val_000000{03,18,30,32,50}.png  dataset/AED20K/segmap_test

    echo '{(0,): 0, (6,): 1, (43,): 2, (1,): 3, (150,): 4, (88,): 5, (97,): 6, (44,): 7, (13,): 8, (5,): 9, (4,): 10, (126,): 11, (105,): 12, (18,): 13, (33,): 14, (139,): 15, (32,): 16, (83,): 17, (9,): 18, (144,): 19, (15,): 20, (20,): 21, (68,): 22, (116,): 23, (149,): 24, (3,): 25, (69,): 26, (14,): 27, (22,): 28, (35,): 29, (120,): 30, (38,): 31, (19,): 32, (48,): 33, (135,): 34, (28,): 35, (64,): 36, (59,): 37, (23,): 38, (82,): 39, (71,): 40, (42,): 41, (66,): 42, (11,): 43, (146,): 44, (16,): 45, (67,): 46, (136,): 47, (99,): 48, (113,): 49, (147,): 50, (25,): 51, (143,): 52, (37,): 53, (138,): 54, (29,): 55, (148,): 56, (86,): 57, (122,): 58, (90,): 59, (54,): 60, (45,): 61, (70,): 62, (111,): 63, (96,): 64, (36,): 65, (39,): 66, (40,): 67, (17,): 68, (93,): 69, (101,): 70, (27,): 71, (47,): 72, (8,): 73, (58,): 74, (31,): 75, (98,): 76, (76,): 77, (34,): 78, (132,): 79, (140,): 80, (65,): 81, (24,): 82, (107,): 83, (10,): 84, (26,): 85, (50,): 86, (63,): 87, (109,): 88, (2,): 89, (118,): 90, (133,): 91, (75,): 92, (77,): 93, (62,): 94, (61,): 95, (127,): 96, (60,): 97, (128,): 98, (12,): 99, (53,): 100, (7,): 101, (131,): 102, (124,): 103, (137,): 104, (81,): 105, (87,): 106, (84,): 107, (103,): 108, (21,): 109, (94,): 110, (30,): 111, (89,): 112, (125,): 113, (119,): 114, (51,): 115, (130,): 116, (72,): 117, (134,): 118, (74,): 119, (142,): 120, (121,): 121, (46,): 122, (104,): 123, (95,): 124, (110,): 125, (56,): 126, (78,): 127, (79,): 128, (73,): 129, (117,): 130, (57,): 131, (102,): 132, (92,): 133, (91,): 134, (55,): 135, (106,): 136, (145,): 137, (41,): 138, (85,): 139, (52,): 140, (115,): 141, (49,): 142, (141,): 143, (112,): 144, (123,): 145, (100,): 146, (80,): 147, (108,): 148, (114,): 149, (129,): 150}' >  dataset/AED20K/segmap_label.txt