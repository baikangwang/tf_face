"""
1 loops all lines from valid.txt
2 split each line by character '|', the third token is the class name, the last token is the index
3 push the distinguish class line in format "\t{index} : '{class name}'," into a list
  Note: the last element without the trail ','
4 write the list to output py file with header and footer,
header: "CLASSES = {"
footer: "}"
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: build_classes.py <input_file>")
        exit(0)

    classes = list()

    with open(sys.argv[1]) as f:
        # 1 loops all lines from valid.txt
        for line in f:
            # 2 split each line by character '|', the third token is the class name, the last token is the index
            tokens = line.split('|')
            image_path = tokens[0]
            index = tokens[1][:-1]  # the last character is the \n
            class_name = image_path.split('/')[2]
            # 3 push the distinguish class line in format "\t{index} : '{class name}'," into a list
            class_line = "\t{0} : '{1}',".format(index, class_name)
            if len(classes)==0 or classes[len(classes)-1] !=class_line:
                classes.append(class_line)

    # 4 write the list to output py file with header and footer,
    with open('classes.py', 'w') as of:
        # add header
        header="CLASSES = {"
        of.write(header+"\n")
        print(header)
        # add list
        for i, line in enumerate(classes):
            strline=line
            if i == len(classes) - 1:
                strline=line[:-1]
            of.write(strline + "\n")
            print(strline)

        # add footer
        footer="}"
        of.write("}")
        print(footer)
