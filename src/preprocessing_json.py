import json


def preprocess_json(json_file):
	with open(json_file, "r", encoding='utf-8') as data_file:
		data = json.load(data_file)

	dict_of_all_marks = {}
	dict_text_id = {}
	for text in data:
		dict_text_id[text["questionId"]] = text["text"]

		question_id = text["questionId"]
		if dict_of_all_marks.get(question_id) is None:
			dict_of_all_marks[question_id] = {}

		dict_of_all_marks[question_id][text["userId"]] = set()
		for triple in text["entities"]:
			dict_of_all_marks[question_id][text["userId"]].add(tuple(triple.values()))

	result = []
	for text_id, text in dict_text_id.items():
		result.append((text, dict_of_all_marks[text_id]))

	return result
