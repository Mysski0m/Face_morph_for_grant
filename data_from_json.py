import json


with open("config.json", encoding="UTF-8") as file_in:
    data = json.load(file_in)

PATH_TO_SAVE_RESULT = data["path_to_save_result"]
PATH_TO_IMPORT_IMAGES = data["path_to_import_images"]
ADD_PRCNT_PPT_Y_UPPER = float(data["add_prcnt_ppt_y_upper"])
ADD_PRCNT_PPT_Y_DOWN = float(data["add_prcnt_ppt_y_down"])
ADD_PRCNT_PPT_X = float(data["add_prcnt_ppt_x"])
DEFAULT_ALPHA_VALUE_FLG = data["enable_default_alpha_value"]
DEFAULT_ALPHA_VALUE = float(data["default_alpha_value"])
COUNT_OF_OUT_MORPHS = data["count_of_out_morphs"]
predictor_model = "./shape_predictor_68_face_landmarks.dat"