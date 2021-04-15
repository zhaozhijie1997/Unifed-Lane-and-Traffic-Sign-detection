# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
# culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
#culane_row_anchor = [239, 244, 249, 253, 258, 263, 268, 272, 277, 282, 287, 292, 296, 301, 306, 311, 316, 321]
# culane_row_anchor = [303,312,322,332,342,353,364,374,383,392,401,411,420,431,441,451,460,469]
# culane_row_anchor = [194,209,224,239,254,269,284,299,314,339,354,369,384,409,424,439,454,469]
culane_row_anchor = [299,309,319,329,339,349,359,369,379,389,399,409,419,429,439,449,459,469]
# 