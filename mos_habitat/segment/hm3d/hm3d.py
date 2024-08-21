# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from nicr_scene_analysis_datasets.dataset_base import (
    DepthStats,
    SceneLabel,
    SceneLabelList,
    SemanticLabel,
    SemanticLabelList,
)
from nicr_scene_analysis_datasets.utils.img import get_colormap


class Hm3dMeta:
    SPLITS = ("train", "val", "test")

    # note that mean and std differ depending on the selected depth_mode
    # however, the impact is marginal, therefore, we decided to use the
    # stats for refined depth for both cases
    # see: my_dataset.depth_compute_stats() for calculation
    TRAIN_SPLIT_DEPTH_STATS = DepthStats(
        min=500.0, max=5000.0, mean=1917.43, std=1069.5
    )
    DEPTH_MODES = ("raw",)
    _DATA_SAMPLE_KEYS = ("identifier", "rgb", "depth")
    _ANNOTATION_SAMPLE_KEYS = ("semantic", "instance")
    SPLIT_SAMPLE_KEYS = {
        SPLITS[0]: _DATA_SAMPLE_KEYS + _ANNOTATION_SAMPLE_KEYS,
        SPLITS[1]: _DATA_SAMPLE_KEYS + _ANNOTATION_SAMPLE_KEYS,
    }
    CAMERAS = ("kv1",)

    DEPTH_DIR = "depth"
    RGB_DIR = "rgb"
    INSTANCES_DIR = "instance"
    NORMAL_DIR = "normal"
    SEMANTIC_DIR_FMT = "semantic_{:d}"
    SEMANTIC_COLORED_DIR_FMT = "semantic_{:d}_colored"

    # semantic labels
    # number of classes without void
    SEMANTIC_N_CLASSES = (81, 53, 33, 23)

    SEMANTIC_CLASS_COLORS_81 = tuple(tuple(c) for c in get_colormap(1 + 81))

    SEMANTIC_LABEL_LIST_81 = SemanticLabelList(
        [
            SemanticLabel("void", False, False, SEMANTIC_CLASS_COLORS_81[0]),
            SemanticLabel("wall", False, False, SEMANTIC_CLASS_COLORS_81[1]),
            SemanticLabel("floor", False, False, SEMANTIC_CLASS_COLORS_81[2]),
            SemanticLabel("chair", True, False, SEMANTIC_CLASS_COLORS_81[3]),
            SemanticLabel("door", True, False, SEMANTIC_CLASS_COLORS_81[4]),
            SemanticLabel("table", True, False, SEMANTIC_CLASS_COLORS_81[5]),
            SemanticLabel("picture", True, False, SEMANTIC_CLASS_COLORS_81[6]),
            SemanticLabel("cabinet", True, False, SEMANTIC_CLASS_COLORS_81[7]),
            SemanticLabel("cushion", True, False, SEMANTIC_CLASS_COLORS_81[8]),
            SemanticLabel("window", True, False, SEMANTIC_CLASS_COLORS_81[9]),
            SemanticLabel("sofa", True, False, SEMANTIC_CLASS_COLORS_81[10]),
            SemanticLabel("bed", True, False, SEMANTIC_CLASS_COLORS_81[11]),
            SemanticLabel("curtain", True, False, SEMANTIC_CLASS_COLORS_81[12]),
            SemanticLabel(
                "chest_of_drawers", True, False, SEMANTIC_CLASS_COLORS_81[13]
            ),
            SemanticLabel("plant", True, False, SEMANTIC_CLASS_COLORS_81[14]),
            SemanticLabel("sink", True, False, SEMANTIC_CLASS_COLORS_81[15]),
            SemanticLabel("stairs", True, False, SEMANTIC_CLASS_COLORS_81[16]),
            SemanticLabel("ceiling", False, False, SEMANTIC_CLASS_COLORS_81[17]),
            SemanticLabel("toilet", True, False, SEMANTIC_CLASS_COLORS_81[18]),
            SemanticLabel("stool", True, False, SEMANTIC_CLASS_COLORS_81[19]),
            SemanticLabel("towel", True, False, SEMANTIC_CLASS_COLORS_81[20]),
            SemanticLabel("mirror", True, False, SEMANTIC_CLASS_COLORS_81[21]),
            SemanticLabel("tv_monitor", True, False, SEMANTIC_CLASS_COLORS_81[22]),
            SemanticLabel("shower", True, False, SEMANTIC_CLASS_COLORS_81[23]),
            SemanticLabel("column", True, False, SEMANTIC_CLASS_COLORS_81[24]),
            SemanticLabel("bathtub", True, False, SEMANTIC_CLASS_COLORS_81[25]),
            SemanticLabel("counter", True, False, SEMANTIC_CLASS_COLORS_81[26]),
            SemanticLabel("fireplace", True, False, SEMANTIC_CLASS_COLORS_81[27]),
            SemanticLabel("lighting", True, False, SEMANTIC_CLASS_COLORS_81[28]),
            SemanticLabel("beam", True, False, SEMANTIC_CLASS_COLORS_81[29]),
            SemanticLabel("railing", True, False, SEMANTIC_CLASS_COLORS_81[30]),
            SemanticLabel("shelving", True, False, SEMANTIC_CLASS_COLORS_81[31]),
            SemanticLabel("blinds", True, False, SEMANTIC_CLASS_COLORS_81[32]),
            SemanticLabel("gym_equipment", True, False, SEMANTIC_CLASS_COLORS_81[33]),
            SemanticLabel("seating", True, False, SEMANTIC_CLASS_COLORS_81[34]),
            SemanticLabel("board_panel", True, False, SEMANTIC_CLASS_COLORS_81[35]),
            SemanticLabel("furniture", True, False, SEMANTIC_CLASS_COLORS_81[36]),
            SemanticLabel("appliances", True, False, SEMANTIC_CLASS_COLORS_81[37]),
            SemanticLabel("clothes", True, False, SEMANTIC_CLASS_COLORS_81[38]),
            SemanticLabel("objects", True, False, SEMANTIC_CLASS_COLORS_81[39]),
            SemanticLabel("misc", True, False, SEMANTIC_CLASS_COLORS_81[40]),
            SemanticLabel("unlabeled", True, False, SEMANTIC_CLASS_COLORS_81[41]),
            # objects for search
            SemanticLabel("bag_of_chips", True, False, SEMANTIC_CLASS_COLORS_81[42]),
            SemanticLabel("bowl", True, False, SEMANTIC_CLASS_COLORS_81[43]),
            SemanticLabel("brush", True, False, SEMANTIC_CLASS_COLORS_81[44]),
            SemanticLabel("calculator", True, False, SEMANTIC_CLASS_COLORS_81[45]),
            SemanticLabel("camera", True, False, SEMANTIC_CLASS_COLORS_81[46]),
            SemanticLabel("can_opener", True, False, SEMANTIC_CLASS_COLORS_81[47]),
            SemanticLabel("candles", True, False, SEMANTIC_CLASS_COLORS_81[48]),
            SemanticLabel("cellphone", True, False, SEMANTIC_CLASS_COLORS_81[49]),
            SemanticLabel("chess_board", True, False, SEMANTIC_CLASS_COLORS_81[50]),
            SemanticLabel("clock", True, False, SEMANTIC_CLASS_COLORS_81[51]),
            SemanticLabel("coat_hanger", True, False, SEMANTIC_CLASS_COLORS_81[52]),
            SemanticLabel("coffee mug", True, False, SEMANTIC_CLASS_COLORS_81[53]),
            SemanticLabel("coke_can", True, False, SEMANTIC_CLASS_COLORS_81[54]),
            SemanticLabel("corkscrew", True, False, SEMANTIC_CLASS_COLORS_81[55]),
            SemanticLabel("deck_of_cards", True, False, SEMANTIC_CLASS_COLORS_81[56]),
            SemanticLabel("drill", True, False, SEMANTIC_CLASS_COLORS_81[57]),
            SemanticLabel("dustpan", True, False, SEMANTIC_CLASS_COLORS_81[58]),
            SemanticLabel("eyeglasses", True, False, SEMANTIC_CLASS_COLORS_81[59]),
            SemanticLabel("flashlight", True, False, SEMANTIC_CLASS_COLORS_81[60]),
            SemanticLabel("flip_flops", True, False, SEMANTIC_CLASS_COLORS_81[61]),
            SemanticLabel("fork", True, False, SEMANTIC_CLASS_COLORS_81[62]),
            SemanticLabel("glass", True, False, SEMANTIC_CLASS_COLORS_81[63]),
            SemanticLabel("handbag", True, False, SEMANTIC_CLASS_COLORS_81[64]),
            SemanticLabel("headphones", True, False, SEMANTIC_CLASS_COLORS_81[65]),
            SemanticLabel("highlighter", True, False, SEMANTIC_CLASS_COLORS_81[66]),
            SemanticLabel("knife", True, False, SEMANTIC_CLASS_COLORS_81[67]),
            SemanticLabel("magnifier", True, False, SEMANTIC_CLASS_COLORS_81[68]),
            SemanticLabel("moka", True, False, SEMANTIC_CLASS_COLORS_81[69]),
            SemanticLabel("mug", True, False, SEMANTIC_CLASS_COLORS_81[70]),
            SemanticLabel("notebook", True, False, SEMANTIC_CLASS_COLORS_81[71]),
            SemanticLabel("pan", True, False, SEMANTIC_CLASS_COLORS_81[72]),
            SemanticLabel("pedestal_fan", True, False, SEMANTIC_CLASS_COLORS_81[73]),
            SemanticLabel("piano_keyboard", True, False, SEMANTIC_CLASS_COLORS_81[74]),
            SemanticLabel("plate", True, False, SEMANTIC_CLASS_COLORS_81[75]),
            SemanticLabel("printer", True, False, SEMANTIC_CLASS_COLORS_81[76]),
            SemanticLabel("rubiks_cube", True, False, SEMANTIC_CLASS_COLORS_81[77]),
            SemanticLabel("scissors", True, False, SEMANTIC_CLASS_COLORS_81[78]),
            SemanticLabel("sharpener", True, False, SEMANTIC_CLASS_COLORS_81[79]),
            SemanticLabel("shoes", True, False, SEMANTIC_CLASS_COLORS_81[80]),
            SemanticLabel("shredder", True, False, SEMANTIC_CLASS_COLORS_81[81]),
        ]
    )
    SEMANTIC_CLASS_COLORS_53 = tuple(tuple(c) for c in get_colormap(1 + 53))

    SEMANTIC_LABEL_LIST_53 = SemanticLabelList(
        [
            SemanticLabel("void", False, False, SEMANTIC_CLASS_COLORS_53[0]),
            SemanticLabel("bed", True, False, SEMANTIC_CLASS_COLORS_53[1]),
            SemanticLabel("books", True, False, SEMANTIC_CLASS_COLORS_53[2]),
            SemanticLabel("ceiling", False, False, SEMANTIC_CLASS_COLORS_53[3]),
            SemanticLabel("chair", True, False, SEMANTIC_CLASS_COLORS_53[4]),
            SemanticLabel("floor", False, False, SEMANTIC_CLASS_COLORS_53[5]),
            SemanticLabel("furniture", True, False, SEMANTIC_CLASS_COLORS_53[6]),
            SemanticLabel("objects", True, False, SEMANTIC_CLASS_COLORS_53[7]),
            SemanticLabel("picture", True, False, SEMANTIC_CLASS_COLORS_53[8]),
            SemanticLabel("sofa", True, False, SEMANTIC_CLASS_COLORS_53[9]),
            SemanticLabel("table", True, False, SEMANTIC_CLASS_COLORS_53[10]),
            SemanticLabel("tv", True, False, SEMANTIC_CLASS_COLORS_53[11]),
            SemanticLabel("wall", False, False, SEMANTIC_CLASS_COLORS_53[12]),
            SemanticLabel("window", True, False, SEMANTIC_CLASS_COLORS_53[13]),
            # objects for search
            SemanticLabel("bag_of_chips", True, False, SEMANTIC_CLASS_COLORS_53[14]),
            SemanticLabel("bowl", True, False, SEMANTIC_CLASS_COLORS_53[15]),
            SemanticLabel("brush", True, False, SEMANTIC_CLASS_COLORS_53[16]),
            SemanticLabel("calculator", True, False, SEMANTIC_CLASS_COLORS_53[17]),
            SemanticLabel("camera", True, False, SEMANTIC_CLASS_COLORS_53[18]),
            SemanticLabel("can_opener", True, False, SEMANTIC_CLASS_COLORS_53[19]),
            SemanticLabel("candles", True, False, SEMANTIC_CLASS_COLORS_53[20]),
            SemanticLabel("cellphone", True, False, SEMANTIC_CLASS_COLORS_53[21]),
            SemanticLabel("chess_board", True, False, SEMANTIC_CLASS_COLORS_53[22]),
            SemanticLabel("clock", True, False, SEMANTIC_CLASS_COLORS_53[23]),
            SemanticLabel("coat_hanger", True, False, SEMANTIC_CLASS_COLORS_53[24]),
            SemanticLabel("coffee mug", True, False, SEMANTIC_CLASS_COLORS_53[25]),
            SemanticLabel("coke_can", True, False, SEMANTIC_CLASS_COLORS_53[26]),
            SemanticLabel("corkscrew", True, False, SEMANTIC_CLASS_COLORS_53[27]),
            SemanticLabel("deck_of_cards", True, False, SEMANTIC_CLASS_COLORS_53[28]),
            SemanticLabel("drill", True, False, SEMANTIC_CLASS_COLORS_53[29]),
            SemanticLabel("dustpan", True, False, SEMANTIC_CLASS_COLORS_53[30]),
            SemanticLabel("eyeglasses", True, False, SEMANTIC_CLASS_COLORS_53[31]),
            SemanticLabel("flashlight", True, False, SEMANTIC_CLASS_COLORS_53[32]),
            SemanticLabel("flip_flops", True, False, SEMANTIC_CLASS_COLORS_53[33]),
            SemanticLabel("fork", True, False, SEMANTIC_CLASS_COLORS_53[34]),
            SemanticLabel("glass", True, False, SEMANTIC_CLASS_COLORS_53[35]),
            SemanticLabel("handbag", True, False, SEMANTIC_CLASS_COLORS_53[36]),
            SemanticLabel("headphones", True, False, SEMANTIC_CLASS_COLORS_53[37]),
            SemanticLabel("highlighter", True, False, SEMANTIC_CLASS_COLORS_53[38]),
            SemanticLabel("knife", True, False, SEMANTIC_CLASS_COLORS_53[39]),
            SemanticLabel("magnifier", True, False, SEMANTIC_CLASS_COLORS_53[40]),
            SemanticLabel("moka", True, False, SEMANTIC_CLASS_COLORS_53[41]),
            SemanticLabel("mug", True, False, SEMANTIC_CLASS_COLORS_53[42]),
            SemanticLabel("notebook", True, False, SEMANTIC_CLASS_COLORS_53[43]),
            SemanticLabel("pan", True, False, SEMANTIC_CLASS_COLORS_53[44]),
            SemanticLabel("pedestal_fan", True, False, SEMANTIC_CLASS_COLORS_53[45]),
            SemanticLabel("piano_keyboard", True, False, SEMANTIC_CLASS_COLORS_53[46]),
            SemanticLabel("plate", True, False, SEMANTIC_CLASS_COLORS_53[47]),
            SemanticLabel("printer", True, False, SEMANTIC_CLASS_COLORS_53[48]),
            SemanticLabel("rubiks_cube", True, False, SEMANTIC_CLASS_COLORS_53[49]),
            SemanticLabel("scissors", True, False, SEMANTIC_CLASS_COLORS_53[50]),
            SemanticLabel("sharpener", True, False, SEMANTIC_CLASS_COLORS_53[51]),
            SemanticLabel("shoes", True, False, SEMANTIC_CLASS_COLORS_53[52]),
            SemanticLabel("shredder", True, False, SEMANTIC_CLASS_COLORS_53[53]),
        ]
    )
    SEMANTIC_CLASS_COLORS_33 = tuple(tuple(c) for c in get_colormap(1 + 33))
    SEMANTIC_LABEL_LIST_33 = SemanticLabelList(
        [
            SemanticLabel("void", False, False, SEMANTIC_CLASS_COLORS_33[0]),
            SemanticLabel("bed", True, False, SEMANTIC_CLASS_COLORS_33[1]),
            SemanticLabel("books", True, False, SEMANTIC_CLASS_COLORS_33[2]),
            SemanticLabel("ceiling", False, False, SEMANTIC_CLASS_COLORS_33[3]),
            SemanticLabel("chair", True, False, SEMANTIC_CLASS_COLORS_33[4]),
            SemanticLabel("floor", False, False, SEMANTIC_CLASS_COLORS_33[5]),
            SemanticLabel("furniture", True, False, SEMANTIC_CLASS_COLORS_33[6]),
            SemanticLabel("objects", True, False, SEMANTIC_CLASS_COLORS_33[7]),
            SemanticLabel("picture", True, False, SEMANTIC_CLASS_COLORS_33[8]),
            SemanticLabel("sofa", True, False, SEMANTIC_CLASS_COLORS_33[9]),
            SemanticLabel("table", True, False, SEMANTIC_CLASS_COLORS_33[10]),
            SemanticLabel("tv", True, False, SEMANTIC_CLASS_COLORS_33[11]),
            SemanticLabel("wall", False, False, SEMANTIC_CLASS_COLORS_33[12]),
            SemanticLabel("window", True, False, SEMANTIC_CLASS_COLORS_33[13]),
            # objects for search
            SemanticLabel("bowl", True, False, SEMANTIC_CLASS_COLORS_33[14]),
            SemanticLabel("brush", True, False, SEMANTIC_CLASS_COLORS_33[15]),
            SemanticLabel("camera", True, False, SEMANTIC_CLASS_COLORS_33[16]),
            SemanticLabel("candles", True, False, SEMANTIC_CLASS_COLORS_33[17]),
            SemanticLabel("cellphone", True, False, SEMANTIC_CLASS_COLORS_33[18]),
            SemanticLabel("chess_board", True, False, SEMANTIC_CLASS_COLORS_33[19]),
            SemanticLabel("coffee mug", True, False, SEMANTIC_CLASS_COLORS_33[20]),
            SemanticLabel("drill", True, False, SEMANTIC_CLASS_COLORS_33[21]),
            SemanticLabel("dustpan", True, False, SEMANTIC_CLASS_COLORS_33[22]),
            SemanticLabel("flip_flops", True, False, SEMANTIC_CLASS_COLORS_33[23]),
            SemanticLabel("glass", True, False, SEMANTIC_CLASS_COLORS_33[24]),
            SemanticLabel("headphones", True, False, SEMANTIC_CLASS_COLORS_33[25]),
            SemanticLabel("magnifier", True, False, SEMANTIC_CLASS_COLORS_33[26]),
            SemanticLabel("moka", True, False, SEMANTIC_CLASS_COLORS_33[27]),
            SemanticLabel("mug", True, False, SEMANTIC_CLASS_COLORS_33[28]),
            SemanticLabel("pedestal_fan", True, False, SEMANTIC_CLASS_COLORS_33[29]),
            SemanticLabel("plate", True, False, SEMANTIC_CLASS_COLORS_33[30]),
            SemanticLabel("rubiks_cube", True, False, SEMANTIC_CLASS_COLORS_33[31]),
            SemanticLabel("shoes", True, False, SEMANTIC_CLASS_COLORS_33[32]),
            SemanticLabel("shredder", True, False, SEMANTIC_CLASS_COLORS_33[33]),
        ]
    )
    
    SEMANTIC_CLASS_COLORS_41 = tuple(tuple(c) for c in get_colormap(1 + 41))
    SEMANTIC_LABEL_LIST_41 = SemanticLabelList([
        SemanticLabel("void", False, False, SEMANTIC_CLASS_COLORS_41[0]),
        SemanticLabel("wall", False, False, SEMANTIC_CLASS_COLORS_41[1]),
        SemanticLabel("floor", False, False, SEMANTIC_CLASS_COLORS_41[2]),
        SemanticLabel("chair", True, False, SEMANTIC_CLASS_COLORS_41[3]),
        SemanticLabel("door", True, False, SEMANTIC_CLASS_COLORS_41[4]),
        SemanticLabel("table", True, False, SEMANTIC_CLASS_COLORS_41[5]),
        SemanticLabel("picture", True, False, SEMANTIC_CLASS_COLORS_41[6]),
        SemanticLabel("cabinet", True, False, SEMANTIC_CLASS_COLORS_41[7]),
        SemanticLabel("cushion", True, False, SEMANTIC_CLASS_COLORS_41[8]),
        SemanticLabel("window", True, False, SEMANTIC_CLASS_COLORS_41[9]),
        SemanticLabel("sofa", True, False, SEMANTIC_CLASS_COLORS_41[10]),
        SemanticLabel("bed", True, False, SEMANTIC_CLASS_COLORS_41[11]),
        SemanticLabel("curtain", True, False, SEMANTIC_CLASS_COLORS_41[12]),
        SemanticLabel("chest_of_drawers", True, False, SEMANTIC_CLASS_COLORS_41[13]),
        SemanticLabel("plant", True, False, SEMANTIC_CLASS_COLORS_41[14]),
        SemanticLabel("sink", True, False, SEMANTIC_CLASS_COLORS_41[15]),
        SemanticLabel("stairs", True, False, SEMANTIC_CLASS_COLORS_41[16]),
        SemanticLabel("ceiling", False, False, SEMANTIC_CLASS_COLORS_41[17]),
        SemanticLabel("toilet", True, False, SEMANTIC_CLASS_COLORS_41[18]),
        SemanticLabel("stool", True, False, SEMANTIC_CLASS_COLORS_41[19]),
        SemanticLabel("towel", True, False, SEMANTIC_CLASS_COLORS_41[20]),
        SemanticLabel("mirror", True, False, SEMANTIC_CLASS_COLORS_41[21]),
        SemanticLabel("tv_monitor", True, False, SEMANTIC_CLASS_COLORS_41[22]),
        SemanticLabel("shower", True, False, SEMANTIC_CLASS_COLORS_41[23]),
        SemanticLabel("column", True, False, SEMANTIC_CLASS_COLORS_41[24]),
        SemanticLabel("bathtub", True, False, SEMANTIC_CLASS_COLORS_41[25]),
        SemanticLabel("counter", True, False, SEMANTIC_CLASS_COLORS_41[26]),
        SemanticLabel("fireplace", True, False, SEMANTIC_CLASS_COLORS_41[27]),
        SemanticLabel("lighting", True, False, SEMANTIC_CLASS_COLORS_41[28]),
        SemanticLabel("beam", True, False, SEMANTIC_CLASS_COLORS_41[29]),
        SemanticLabel("railing", True, False, SEMANTIC_CLASS_COLORS_41[30]),
        SemanticLabel("shelving", True, False, SEMANTIC_CLASS_COLORS_41[31]),
        SemanticLabel("blinds", True, False, SEMANTIC_CLASS_COLORS_41[32]),
        SemanticLabel("gym_equipment", True, False, SEMANTIC_CLASS_COLORS_41[33]),
        SemanticLabel("seating", True, False, SEMANTIC_CLASS_COLORS_41[34]),
        SemanticLabel("board_panel", True, False, SEMANTIC_CLASS_COLORS_41[35]),
        SemanticLabel("furniture", True, False, SEMANTIC_CLASS_COLORS_41[36]),
        SemanticLabel("appliances", True, False, SEMANTIC_CLASS_COLORS_41[37]),
        SemanticLabel("clothes", True, False, SEMANTIC_CLASS_COLORS_41[38]),
        SemanticLabel("objects", True, False, SEMANTIC_CLASS_COLORS_41[39]),
        SemanticLabel("misc", True, False, SEMANTIC_CLASS_COLORS_41[40]),
        SemanticLabel("unlabeled", False, False, SEMANTIC_CLASS_COLORS_41[41]),    
    ])

    SEMANTIC_CLASS_COLORS_22 = tuple(tuple(c) for c in get_colormap(1 + 22))
    SEMANTIC_LABEL_LIST_22 = SemanticLabelList(
        [
            SemanticLabel("unexplored", False, False, SEMANTIC_CLASS_COLORS_22[0]),
            SemanticLabel("unoccupied", False, False, SEMANTIC_CLASS_COLORS_22[1]),
            SemanticLabel("occupied", False, False, SEMANTIC_CLASS_COLORS_22[2]),
            # objects for search
            SemanticLabel("bowl", True, False, SEMANTIC_CLASS_COLORS_22[3]),
            SemanticLabel("brush", True, False, SEMANTIC_CLASS_COLORS_22[4]),
            SemanticLabel("camera", True, False, SEMANTIC_CLASS_COLORS_22[5]),
            SemanticLabel("candles", True, False, SEMANTIC_CLASS_COLORS_22[6]),
            SemanticLabel("cellphone", True, False, SEMANTIC_CLASS_COLORS_22[7]),
            SemanticLabel("chess_board", True, False, SEMANTIC_CLASS_COLORS_22[8]),
            SemanticLabel("coffee mug", True, False, SEMANTIC_CLASS_COLORS_22[9]),
            SemanticLabel("drill", True, False, SEMANTIC_CLASS_COLORS_22[10]),
            SemanticLabel("dustpan", True, False, SEMANTIC_CLASS_COLORS_22[11]),
            SemanticLabel("flip_flops", True, False, SEMANTIC_CLASS_COLORS_22[12]),
            SemanticLabel("glass", True, False, SEMANTIC_CLASS_COLORS_22[13]),
            SemanticLabel("headphones", True, False, SEMANTIC_CLASS_COLORS_22[14]),
            SemanticLabel("magnifier", True, False, SEMANTIC_CLASS_COLORS_22[15]),
            SemanticLabel("moka", True, False, SEMANTIC_CLASS_COLORS_22[16]),
            SemanticLabel("mug", True, False, SEMANTIC_CLASS_COLORS_22[17]),
            SemanticLabel("pedestal_fan", True, False, SEMANTIC_CLASS_COLORS_22[18]),
            SemanticLabel("plate", True, False, SEMANTIC_CLASS_COLORS_22[19]),
            SemanticLabel("rubiks_cube", True, False, SEMANTIC_CLASS_COLORS_22[20]),
            SemanticLabel("shoes", True, False, SEMANTIC_CLASS_COLORS_22[21]),
            SemanticLabel("shredder", True, False, SEMANTIC_CLASS_COLORS_22[22]),
        ]
    )


if __name__ == "__main__":
    import cv2
    import numpy as np

    # Assume that the definition of the class SemanticLabel is given and the above lists are defined
    # Number of columns
    columns = 2

    # Create a blank white image
    # Change height and width accordingly
    height = 50 * ((len(Hm3dMeta.SEMANTIC_LABEL_LIST_33) // columns) + 1)
    width = 500 * columns  # width is adjusted according to the number of columns
    image = np.ones((height, width, 3), np.uint8) * 255

    # Use cv2.putText to write the label names into the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # Black
    line_type = 2

    for i, item in enumerate(Hm3dMeta.SEMANTIC_LABEL_LIST_33):
        # Draw the color next to the text as a filled rectangle
        bgr = tuple(item.color[::-1])
        # Calculate column and row number
        col = i // ((len(Hm3dMeta.SEMANTIC_LABEL_LIST_33) // columns) + 1)
        row = i % ((len(Hm3dMeta.SEMANTIC_LABEL_LIST_33) // columns) + 1)

        # Coordinates for the rectangle and text
        rect_start = (col * 500 + 10, row * 50 + 10)
        rect_end = (col * 500 + 40, row * 50 + 40)
        text_pos = (col * 500 + 50, row * 50 + 30)

        # Draw the color next to the text as a filled rectangle
        cv2.rectangle(
            image, rect_start, rect_end, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1
        )

        # Put the label text
        cv2.putText(
            image,
            item.class_name,
            text_pos,  # Position
            font,
            font_scale,
            font_color,
            line_type,
        )

    # Save the image
    cv2.imwrite("legend.png", image)
