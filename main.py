import argparse
from src.solution import Solution
from src.prepare_data import print_text_with_labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="txt file with input text")
	parser.add_argument("-o", "--output", help="txt file for output text")
	parser.add_argument("-d", "--debug", action="store_true", help="turn in debug output")
	args = parser.parse_args()

	with open(args.input, "r", encoding="utf-8") as input_file:
		input_text = input_file.read()

	solution = Solution(model_path="resources/model.out")
	set_of_labels = solution.predict([input_text], debug=args.debug)[0]

	with open(args.output, "w+") as output_file:
		res = print_text_with_labels(input_text,set_of_labels)
		output_file.write(res)
		if args.debug:
			print(res)

